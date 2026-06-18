#!/usr/bin/env bash
# Evaluate a finetuned streaming model across the chunk-size sweep (EVAL_CTXS in env.sh).
# Prints RNNT + CTC WER per chunk size (compare_vs_offline shows the offline ceiling).
#
# Usage:
#   ./03_eval_sweep.sh <ARM>                 # auto-finds newest finetuned .nemo under exp/ft/<ARM>
#   ./03_eval_sweep.sh <FINETUNED_NEMO>      # explicit .nemo path
#   ./03_eval_sweep.sh <ARM|NEMO> <TEST_MANIFEST>
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

ARG="${1:?usage: 03_eval_sweep.sh <ARM|finetuned.nemo> [TEST_MANIFEST]}"
TEST="${2:-$TEST_MANIFEST}"

if [[ -f "$ARG" ]]; then
  MODEL="$ARG"
else
  MODEL="$(latest_nemo "$EXP_ROOT/ft/$ARG")"
  [[ -n "$MODEL" ]] || { echo "ERROR: no finetuned .nemo under $EXP_ROOT/ft/$ARG; pass a path." >&2; exit 1; }
fi
_require_files NEMO_ROOT
[[ "$TEST" != "???" && -n "$TEST" ]] || { echo "ERROR: TEST_MANIFEST not set (env.sh) and none passed." >&2; exit 1; }

echo ">>> Eval model=$MODEL  test=$TEST"
cd "$NEMO_ROOT"
for CTX in "${EVAL_CTXS[@]}"; do
  echo "================ chunk att_context_size=$CTX ================"
  python examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py \
    model_path="$MODEL" \
    dataset_manifest="$TEST" \
    att_context_size="$CTX" \
    batch_size=16 \
    compare_vs_offline=true
done
