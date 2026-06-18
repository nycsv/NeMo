#!/usr/bin/env bash
# Arm B0: encoder-only streaming BEST-RQ baseline. Extra args pass through as Hydra overrides.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_pretrain_common.sh"
run_pretrain nest_fast-conformer_streaming B0_streaming "$@"
