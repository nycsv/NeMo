#!/bin/bash
set -euo pipefail

PRETRAINED_NAME="salm_results/checkpoints/step=1000.ckpt"  # 또는 "nvidia/canary-qwen-2.5b"
INPUT_MANIFEST="/data/test_manifest.json"
OUTPUT_MANIFEST="generations.jsonl"
BATCH_SIZE=32
MAX_NEW_TOKENS=128
USER_PROMPT="Transcribe the following:"   
DEVICE="cuda"
DTYPE="bfloat16"

echo "[INFO] 생성 시작"
echo "[INFO] 모델  : $PRETRAINED_NAME"
echo "[INFO] 입력  : $INPUT_MANIFEST"
echo "[INFO] 출력  : $OUTPUT_MANIFEST"
echo "[INFO] 프롬프트: $USER_PROMPT"

python examples/speechlm2/salm_generate.py \
    pretrained_name="$PRETRAINED_NAME" \
    inputs="$INPUT_MANIFEST" \
    output_manifest="$OUTPUT_MANIFEST" \
    batch_size="$BATCH_SIZE" \
    max_new_tokens="$MAX_NEW_TOKENS" \
    user_prompt="$USER_PROMPT" \
    device="$DEVICE" \
    dtype="$DTYPE" \
    verbose=true

echo "[INFO] Complted: $OUTPUT_MANIFEST"
