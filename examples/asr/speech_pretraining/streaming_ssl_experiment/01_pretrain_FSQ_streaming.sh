#!/usr/bin/env bash
# Optional later arm: streaming BEST-RQ-style SSL with FSQ target tokenizer (encoder-only streaming).
# To combine FSQ with dual-mode, append:
#   ssl_model_class=masked_token_pred_dualmode +model.dual_mode.alpha=1.0 +model.dual_mode.beta=1.0
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_pretrain_common.sh"
run_pretrain nest_fast-conformer_streaming_fsq FSQ_streaming "$@"
