#!/usr/bin/env bash
# Arm A: OFFLINE (bidirectional) BEST-RQ SSL baseline. Extra args pass through as Hydra overrides.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_pretrain_common.sh"
run_pretrain nest_fast-conformer_offline A_offline "$@"
