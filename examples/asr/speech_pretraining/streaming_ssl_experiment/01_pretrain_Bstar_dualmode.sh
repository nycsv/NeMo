#!/usr/bin/env bash
# Arm B*: PROPOSED streaming BEST-RQ with dual-mode consistency.
# Tune dual-mode weights by appending e.g.  model.dual_mode.beta=0.5
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_pretrain_common.sh"
run_pretrain nest_fast-conformer_dualmode_streaming Bstar_dualmode "$@"
