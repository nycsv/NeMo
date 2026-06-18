#!/usr/bin/env bash
# Build the BPE (SentencePiece) tokenizer for finetuning, from the train-clean-100 manifest.
# Produces $TOKENIZER_DIR (= $TOKENIZER_BUILD_ROOT/tokenizer_spe_bpe_v$TOKENIZER_VOCAB_SIZE),
# which env.sh already points the finetune scripts at. Run this once before 02_finetune.sh.
#
# Extra args pass through to process_asr_text_tokenizer.py, e.g.:
#   ./00_build_tokenizer.sh --spe_character_coverage=1.0
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
_require_files NEMO_ROOT FT_TRAIN_MANIFEST

mkdir -p "$TOKENIZER_BUILD_ROOT"
echo ">>> Building $TOKENIZER_TYPE tokenizer (vocab=$TOKENIZER_VOCAB_SIZE) from $FT_TRAIN_MANIFEST"
cd "$NEMO_ROOT"
set -x
python scripts/tokenizers/process_asr_text_tokenizer.py \
  --manifest="$FT_TRAIN_MANIFEST" \
  --data_root="$TOKENIZER_BUILD_ROOT" \
  --vocab_size="$TOKENIZER_VOCAB_SIZE" \
  --tokenizer=spe \
  --spe_type=bpe \
  "$@"
set +x

if [[ -f "$TOKENIZER_DIR/tokenizer.model" ]]; then
  echo "OK: tokenizer ready at $TOKENIZER_DIR"
else
  echo "WARNING: expected $TOKENIZER_DIR/tokenizer.model not found." >&2
  echo "If you passed --spe_pad/--spe_bos/--spe_eos or --spe_max_sentencepiece_length, the dir name" >&2
  echo "differs; update TOKENIZER_DIR in env.sh to match what was produced under:" >&2
  echo "  $TOKENIZER_BUILD_ROOT" >&2
  ls -1 "$TOKENIZER_BUILD_ROOT" 2>/dev/null || true
fi
