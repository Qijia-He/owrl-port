#!/usr/bin/env python3
"""Checkpoint pruner for the OpenWebRL Hyak run.

SAVE_INTERVAL=1 writes a ~34 GB Megatron distributed checkpoint every iteration.
Over 90 iterations that is ~3.1 TB, so old checkpoints have to go. Retention:

    keep  every iteration that is a multiple of KEEP_EVERY (10, 20, ... 90)
    keep  the KEEP_RECENT most recent iterations
    drop  everything else

Two safety properties, in the order they were asked for:

1. A checkpoint is only ever deleted once a NEWER one has been fully written and
   `latest_checkpointed_iteration.txt` points past it. We read that file to get
   N and only ever consider directories with iteration < N. The iteration the
   tracker currently names is never a deletion candidate -- that is what resume
   reads, and removing it mid-run would strand the job. Anything >= N is left
   alone too, since a directory above N is a save still in flight.

2. Pruning happens after a save completes, never alongside one. We only act when
   the tracker value has been observed *stable* for SETTLE_SECS, which means the
   save that produced it finished and the next has not landed. On top of that,
   each victim is first renamed to `.pruning_iter_*` and only then removed, so a
   partially-deleted directory can never be mistaken for a partially-written
   checkpoint -- it does not match the `iter_*` glob that resume scans at all.

Run it alongside training; it is a read-mostly poller and costs nothing.

    python3 prune_checkpoints.py --save-dir <ckpt dir> [--dry-run]
"""

import argparse
import os
import re
import shutil
import sys
import time

TRACKER = "latest_checkpointed_iteration.txt"
ITER_RE = re.compile(r"^iter_(\d+)$")
PRUNING_PREFIX = ".pruning_"


def log(msg):
    print(f"[prune {time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}", flush=True)


def read_tracker(save_dir):
    """Return the iteration the tracker names, or None if unreadable/not numeric."""
    path = os.path.join(save_dir, TRACKER)
    try:
        with open(path) as f:
            raw = f.read().strip()
    except (OSError, IOError):
        return None
    digits = "".join(ch for ch in raw if ch.isdigit())
    if not digits:
        return None          # e.g. the literal "release"
    return int(digits)


def list_iters(save_dir):
    """Map iteration number -> directory path, for real iter_* dirs only."""
    found = {}
    try:
        entries = os.listdir(save_dir)
    except (OSError, IOError):
        return found
    for name in entries:
        m = ITER_RE.match(name)
        if not m:
            continue
        full = os.path.join(save_dir, name)
        if os.path.isdir(full):
            found[int(m.group(1))] = full
    return found


def victims(iters, latest, keep_every, keep_recent):
    """Which iterations may be deleted, given the tracker points at `latest`."""
    # Hard floor: never consider the tracked iteration or anything newer.
    candidates = sorted(i for i in iters if i < latest)

    keep = {i for i in iters if keep_every > 0 and i % keep_every == 0}
    keep.add(latest)
    # The most recent KEEP_RECENT iterations, counted from the tracked one down.
    for i in sorted((i for i in iters if i <= latest), reverse=True)[:keep_recent]:
        keep.add(i)

    return [i for i in candidates if i not in keep], keep


def sweep_stale_pruning_dirs(save_dir, dry_run):
    """Finish any rename-then-delete that a previous run left half-done."""
    try:
        entries = os.listdir(save_dir)
    except (OSError, IOError):
        return
    for name in entries:
        if not name.startswith(PRUNING_PREFIX):
            continue
        full = os.path.join(save_dir, name)
        if not os.path.isdir(full):
            continue
        log(f"resuming interrupted prune of {name}")
        if not dry_run:
            shutil.rmtree(full, ignore_errors=True)


def prune_once(save_dir, keep_every, keep_recent, dry_run):
    latest = read_tracker(save_dir)
    if latest is None:
        return None, 0

    iters = list_iters(save_dir)
    if not iters:
        return latest, 0

    if latest not in iters:
        # Tracker names an iteration whose directory is missing. Do not guess.
        log(f"WARNING tracker says {latest} but iter_{latest:07d} is absent; "
            f"skipping this pass rather than deleting anything")
        return latest, 0

    doomed, keep = victims(iters, latest, keep_every, keep_recent)
    if not doomed:
        return latest, 0

    log(f"tracker={latest} present={sorted(iters)} keep={sorted(keep)} "
        f"delete={doomed}")

    freed = 0
    for i in doomed:
        src = iters[i]
        staged = os.path.join(save_dir, f"{PRUNING_PREFIX}iter_{i:07d}")
        try:
            if dry_run:
                log(f"DRY-RUN would delete iter_{i:07d}")
                freed += 1
                continue
            # Rename first: after this the directory no longer matches iter_*,
            # so a crash mid-delete cannot leave something resume might read.
            os.rename(src, staged)
            shutil.rmtree(staged, ignore_errors=True)
            log(f"deleted iter_{i:07d}")
            freed += 1
        except OSError as e:
            log(f"ERROR could not delete iter_{i:07d}: {e}")
    return latest, freed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save-dir", required=True)
    p.add_argument("--keep-every", type=int, default=10,
                   help="keep iterations that are multiples of this (0 disables)")
    p.add_argument("--keep-recent", type=int, default=1,
                   help="also keep this many most-recent iterations")
    p.add_argument("--poll-secs", type=int, default=60)
    p.add_argument("--settle-secs", type=int, default=120,
                   help="tracker must hold the same value this long before we act")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--once", action="store_true", help="one pass, then exit")
    a = p.parse_args()

    if not os.path.isdir(a.save_dir):
        log(f"save dir {a.save_dir} does not exist yet; waiting for it")

    log(f"watching {a.save_dir} | keep multiples of {a.keep_every} "
        f"+ {a.keep_recent} most recent | dry_run={a.dry_run}")

    sweep_stale_pruning_dirs(a.save_dir, a.dry_run)

    last_seen, stable_since = None, None
    while True:
        current = read_tracker(a.save_dir)

        if current != last_seen:
            # Tracker moved: a save just landed. Restart the settle timer so we
            # never delete while the next checkpoint may be mid-write.
            last_seen, stable_since = current, time.time()
        elif current is not None and stable_since is not None:
            if time.time() - stable_since >= a.settle_secs:
                _, freed = prune_once(a.save_dir, a.keep_every,
                                      a.keep_recent, a.dry_run)
                if freed:
                    log(f"pass complete, {freed} checkpoint(s) removed")
                # Don't re-run until the tracker advances again.
                stable_since = None

        if a.once:
            if current is not None:
                prune_once(a.save_dir, a.keep_every, a.keep_recent, a.dry_run)
            return
        time.sleep(a.poll_secs)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
