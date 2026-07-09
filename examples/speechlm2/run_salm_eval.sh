#!/bin/bash
set -euo pipefail

PRETRAINED_NAME="salm_results/checkpoints/step=1000.ckpt"
# Required only when PRETRAINED_NAME is a raw training checkpoint (.ckpt file or
# FSDP/TP distributed-checkpoint dir): the exp config whose .model subtree rebuilds
# the architecture. Leave empty ("") for HF Hub IDs or to_hf.py / save_pretrained dirs.
CKPT_CONFIG="salm_results/exp_config.yaml"
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

# ckpt_config는 raw 체크포인트일 때만 필요 (설정된 경우 존재 확인)
if [ -n "$CKPT_CONFIG" ] && [ ! -f "$CKPT_CONFIG" ]; then
    echo "[ERROR]: ckpt_config not found: $CKPT_CONFIG"
    exit 1
fi

echo "[INFO] 평가 시작"
echo "[INFO] 모델: $PRETRAINED_NAME"
echo "[INFO] 입력: $INPUT_MANIFEST"
echo "[INFO] 출력: $OUTPUT_MANIFEST"

# ckpt_config는 raw 체크포인트(.ckpt / dist-ckpt dir)일 때만 전달; HF 모델이면 생략
EXTRA_ARGS=()
if [ -n "$CKPT_CONFIG" ]; then
    EXTRA_ARGS+=( ckpt_config="$CKPT_CONFIG" )
fi

python "examples/speechlm2/salm_eval.py" \
    pretrained_name="$PRETRAINED_NAME" \
    inputs="$INPUT_MANIFEST" \
    batch_size="$BATCH_SIZE" \
    max_new_tokens="$MAX_NEW_TOKENS" \
    output_manifest="$OUTPUT_MANIFEST" \
    use_normalizer="$NORMALIZER" \
    device="$DEVICE" \
    dtype="$DTYPE" \
    verbose=true \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

echo "[INFO] Completed: $OUTPUT_MANIFEST"
