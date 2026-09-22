#!/usr/bin/env bash
# Run the T1..T9 lightweight screening groups sequentially.
# T0 is run separately; this script is launched only after the rich feature
# export has been verified. Logs to screening/screen_groups.log.
set -euo pipefail

export PATH=/root/miniconda3/envs/agiclass/bin:$PATH
cd /root/3eedqa/qa_pipeline
mkdir -p /root/autodl-tmp/3eed_data/screening

for g in T1 T2 T3 T4 T5 T6 T7 T8 T9; do
    echo "=== GROUP $g $(date -u +%FT%TZ) ==="
    python screen_token_groups.py --group "$g" 2>&1 | tail -3
    echo "=== GROUP $g DONE $(date -u +%FT%TZ) ==="
done
echo ALL_GROUPS_DONE
