# Maintaining the experiment record

The [main README](README.md) is the single, self-contained report for readers. This file explains how its tables and supporting records are maintained.

The published location is [`Qijia-He/owrl-port/docs/action-reward-experiments/`](https://github.com/Qijia-He/owrl-port/tree/main/docs/action-reward-experiments) on `main`. The folder contains the report, exported results, and a portable renderer. The full benchmark/collection/training runtime is maintained in the separate `arm-generalization` cluster project.

## Files

| File | Purpose |
|---|---|
| `README.md` | Human-maintained explanation, citations, setup, interpretation, and next steps, with explicitly marked generated numerical blocks |
| `runs/<run-id>.md` | Generated detailed metrics, paired comparisons, diagnostics, CI procedure, costs, and audit |
| `data/<run-id>.json` | Frozen portable result package and source-artifact SHA-256 hashes |
| `diagnostics/<study-id>.json` | Audited bounded-study metrics and hashes, using a separate diagnostic schema |
| `DECISIONS.md` | Execution scope, research decisions, limitations, and operational history |
| `training-data-readiness.md` | Audited state/panel counts, reusable training formats, label diagnostics and remaining feature work |
| `selection-arm-head-plan.md` | Current Table 1(c) experiment plan, method distinctions, adaptation requirements and blockers |
| `literature-length-alignment.md` | Dated paper/code audit of output limits, model identities, reported scores, and protocol differences |
| `tools/update_experiment_log.py` | Portable copy of the cluster project's canonical `scripts/update_experiment_log.py` |

The public package does not contain credentials, raw images, machine-specific paths, or API request payloads. It is sufficient to regenerate the published tables; it is not the full runtime or raw episode release.

The bounded output-length diagnostic uses a different design from the original six-group, four-condition pilot. Its portable metrics live in `diagnostics/`, and its report and main-README diagnostic section are maintained from those audited metrics. Do not put this schema in `data/` or pass it to the original pilot renderer. The renderer still updates only the original pilot's marked blocks; it does not refresh the diagnostic narrative or tables. Keep both schemas and result versions distinct, and preserve the context-exhausted attempt in the diagnostic denominator.

## Render existing results

From the downloaded GitHub documentation directory:

```bash
python3 tools/update_experiment_log.py --docs-root .
```

Only Python's standard library is needed. The command reads exported JSON, regenerates per-run tables, and replaces the `SNAPSHOT`, `SUCCESS`, `REWARD`, `COST`, and `HISTORY` blocks in `README.md`. It preserves prose outside those markers. The existing README must be present with each marker pair exactly once.

From the cluster experiment project root, the equivalent command is:

```bash
python3 scripts/update_experiment_log.py
```

Neither command runs models, spends API credits, submits jobs, or pushes to GitHub.

## Register an audited snapshot

The following command runs in the cluster experiment project, **not** at the `owrl-port` repository root:

```bash
source scripts/runtime.sh
"$ARM_PYTHON" scripts/update_experiment_log.py \
  --register reports/pilot-seed42-artifacts.json \
  --id v3-sol-nativecmd-seed42
```

Registration additionally requires the matching experiment configuration, frozen source artifacts, and PyYAML from the experiment environment. It verifies source-file hashes, the scientific configuration hash, complete task/seed coverage, final audit status, and complete group-sensitivity results. An existing run ID cannot be overwritten with different numerical data.

`scripts/freeze_seed42_report.py` invokes registration after freezing. Future experiments need their own receipt and stable run ID. The existing frozen method contract records the original three-seed plan; the executed-seed fields and audit record the actual scope. Do not alter a frozen contract merely to rewrite historical planning metadata.

## Update checklist for subsequent authorized work

1. Update the scope/status prose when an experiment starts, pauses, or changes. Do not put incomplete runs in an audited-results table.
2. After completion and audit, register the new snapshot under a new ID and regenerate the tables.
3. **Review the whole main README**, including its opening conclusion, setup, interpretation, and next steps. Numeric regeneration cannot update scientific judgments. Readers should still understand the project from that one document.
4. Update `DECISIONS.md` with the evidence, limitations, and user decisions. If the protocol changes, revise the applicable per-run description instead of silently applying the seed-42 setup to a different design.
5. Keep the canonical and portable renderer copies identical. Check links, table/data agreement, idempotent rendering, and preservation of existing result packages.
6. Synchronize the local `docs/experiment-log/` directory to the published directory using the mapping in `configs/experiment-log-publish.json`. Read the latest branch head and use an expected-head check so concurrent changes are preserved.
7. Read back the published files and verify their contents against the reviewed local files.

GitHub synchronization is performed during subsequent work, not by an unattended uploader. Maintaining this record does not itself authorize extra seeds, ARM training, or official test evaluation.

The authorized v6f continuation records exact actor token IDs and preserves all v6e accepted examples. Its Slurm chain freezes top-level `armbench/*.py`, `scripts/*.py`, and `scripts/runtime.sh`; changing those while collection is active causes the next stage to stop. Preserve the source snapshot and create a new study version for runtime changes. Compiler caches are regenerable scratch data; observations, labels, dataset snapshots, and training artifacts remain in project storage. The [v6f validation receipt](diagnostics/v6f-pipeline-validation.json) records pipeline checks only. Do not place their tiny-batch losses or checkpoint counts in benchmark accuracy tables.

The active collection is now `v6g-native-empty-image-text-web-1024-seed42`, inheriting the v6f token/training checks and preserving 2,912 audited states. The same runtime-source freeze applies. `reports/collection_supervisor.py` is separate from the frozen experiment code and has its own recorded SHA-256; do not edit it while its dependency chain is installed. Its current record is `<study>/supervisor/status.json`. The watcher follows afterany dependencies using CPU-only jobs and leaves GPUs unallocated while waiting. To intentionally stop recovery, create `<study>/supervisor/STOP` and cancel the actual work and pending watcher jobs through Slurm. Do not automatically retry cancellation, storage, authentication, OOM or code/data failures. Full retry and restart semantics are in [collection-status.md](collection-status.md).
