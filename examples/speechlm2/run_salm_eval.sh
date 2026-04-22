#!/bin/bash
set -euo pipefail

PRETRAINED_NAME="salm_results/checkpoints/step=1000.ckpt"
INPUT_MANIFEST="/data/val_manifest.json"   # NeMo JSON manifest
OUTPUT_MANIFEST="eval_output.jsonl"
BATCH_SIZE=32
MAX_NEW_TOKENS=128
DEVICE="cuda"
DTYPE="bfloat16"
NORMALIZER="english"   # english | basic | none
if [ ! -f "$PRETRAINED_NAME" ] && [ ! -d "$PRETRAINED_NAME" ]; then
    echo "[ERROR]: $PRETRAINED_NAME"
    exit 1
fi

# 입력 파일 확인
if [ ! -f "$INPUT_MANIFEST" ]; then
    echo "[ERROR]: $INPUT_MANIFEST"
    exit 1
fi

echo "[INFO] 평가 시작"
echo "[INFO] 모델: $PRETRAINED_NAME"
echo "[INFO] 입력: $INPUT_MANIFEST"
echo "[INFO] 출력: $OUTPUT_MANIFEST"

python "examples/speechlm2/salm_eval.py" \
    pretrained_name="$PRETRAINED_NAME" \
    inputs="$INPUT_MANIFEST" \
    batch_size="$BATCH_SIZE" \
    max_new_tokens="$MAX_NEW_TOKENS" \
    output_manifest="$OUTPUT_MANIFEST" \
    use_normalizer="$NORMALIZER" \
    device="$DEVICE" \
    dtype="$DTYPE" \
    verbose=true

echo "[INFO] Completed: $OUTPUT_MANIFEST"
