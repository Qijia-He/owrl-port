#!/usr/bin/env bash
# Run once AFTER stage_from_xunmei.sh. Edits only MY repo copy.
# Screenshot timeout 30 s -> 60 s: 8.2k of Xunmei's 14k sessions died on
# "Page.screenshot: Timeout 30000ms" at /reset. Infrastructure knob (guide §7).
set -euo pipefail
CFG=/gscratch/cse/heqj3/owrl/OpenWebRL/openwebrl/env/config.yaml
grep -q '^screenshot_timeout: 30000' "$CFG" && sed -i 's/^screenshot_timeout: 30000/screenshot_timeout: 60000/' "$CFG"
grep -nE '^(wait_timeout|screenshot_timeout|max_retries):' "$CFG"
