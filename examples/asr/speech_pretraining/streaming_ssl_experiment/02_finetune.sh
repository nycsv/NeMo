#!/usr/bin/env bash
# Finetune a pretrained SSL encoder into a streaming Hybrid RNNT+CTC model on train-clean-100.
#
# Usage:
#   ./02_finetune.sh <ARM> [extra hydra overrides...]
#   <ARM> is one of: A_offline | B0_streaming | Bstar_dualmode | FSQ_streaming
#   By default the newest SSL .nemo under exp/ssl/<ARM> is used. To pick an explicit checkpoint:
#   SSL_NEMO=/path/to/ssl.nemo ./02_finetune.sh <ARM>
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

ARM="${1:?usage: 02_finetune.sh <ARM> [extra hydra overrides...]}"; shift || true
SSL_NEMO="${SSL_NEMO:-}"

if [[ -z "$SSL_NEMO" ]]; then
  SSL_NEMO="$(latest_nemo "$EXP_ROOT/ssl/$ARM")"
  [[ -n "$SSL_NEMO" ]] || { echo "ERROR: no .nemo found under $EXP_ROOT/ssl/$ARM; set SSL_NEMO=..." >&2; exit 1; }
fi
_require_files NEMO_ROOT FT_TRAIN_MANIFEST FT_DEV_MANIFEST TOKENIZER_DIR
[[ -f "$SSL_NEMO" ]] || { echo "ERROR: SSL checkpoint not found: $SSL_NEMO" >&2; exit 1; }

OUT="$EXP_ROOT/ft/$ARM"
mkdir -p "$OUT"
echo ">>> Finetune arm=$ARM  encoder<-$SSL_NEMO  -> $OUT"
cd "$NEMO_ROOT"
set -x
python examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe.py \
  --config-path=../conf/fastconformer/hybrid_cache_aware_streaming \
  --config-name=ssl_streaming_hybrid_finetune \
  init_from_nemo_model.encoder.path="$SSL_NEMO" \
  model.tokenizer.dir="$TOKENIZER_DIR" \
  model.tokenizer.type="$TOKENIZER_TYPE" \
  model.train_ds.manifest_filepath="$FT_TRAIN_MANIFEST" \
  model.validation_ds.manifest_filepath="$FT_DEV_MANIFEST" \
  trainer.devices="$DEVICES" \
  trainer.num_nodes="$NUM_NODES" \
  trainer.precision="$PRECISION" \
  trainer.max_steps="$FT_MAX_STEPS" \
  model.optim.sched.warmup_steps="$FT_WARMUP" \
  exp_manager.exp_dir="$OUT" \
  exp_manager.name="$ARM" \
  "$@"
