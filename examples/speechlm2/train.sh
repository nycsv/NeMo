#!/bin/bash
# =============================================================================
#  run_finetune.sh
#  싱글 노드 멀티 GPU 파인튜닝 실행 스크립트
#  torchrun 사용 (SLURM/srun 불필요)
# =============================================================================

# ------------------------------------------------------------------ #
#  설정값 — 여기만 수정하세요                                           #
# ------------------------------------------------------------------ #

# 사용할 GPU 번호 (예: "0,1,2,3" 또는 "0,1")
CUDA_VISIBLE="0,1,2,3"

# GPU 개수 (CUDA_VISIBLE 개수와 일치시킬 것)
NUM_GPUS=4

# 경로
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEMO_CKPT="${SCRIPT_DIR}/canary-qwen-2.5b.nemo"
TRAIN_INPUT_CFG="${SCRIPT_DIR}/input_cfg.yaml"
VAL_CUTS="/path/to/val_cuts.jsonl.gz"
RESULTS_DIR="${SCRIPT_DIR}/results"
LOG_DIR="${SCRIPT_DIR}/logs"

# ------------------------------------------------------------------ #
#  사전 체크                                                            #
# ------------------------------------------------------------------ #
set -e  # 오류 발생 시 즉시 종료

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

echo "=============================================="
echo "  canary-qwen-2.5b Fine-tuning"
echo "  GPUs : ${CUDA_VISIBLE} (${NUM_GPUS}개)"
echo "  CKPT : ${NEMO_CKPT}"
echo "  결과 : ${RESULTS_DIR}"
echo "=============================================="

# checkpoint 존재 확인
if [ ! -f "${NEMO_CKPT}" ]; then
    echo "[ERROR] NeMo checkpoint 없음: ${NEMO_CKPT}"
    echo "        save_modules.py 또는 HuggingFace Hub에서 먼저 다운로드하세요."
    exit 1
fi

# ------------------------------------------------------------------ #
#  환경 변수                                                            #
# ------------------------------------------------------------------ #
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE}"

# NCCL 설정 (단일 노드 최적화)
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1          # InfiniBand 없는 경우
export NCCL_P2P_DISABLE=0         # NVLink P2P 활성화
export NCCL_SOCKET_IFNAME=lo      # 단일 노드는 loopback 사용

# PyTorch 관련
export TORCH_NCCL_BLOCKING_WAIT=1
export OMP_NUM_THREADS=4

# ------------------------------------------------------------------ #
#  torchrun으로 실행                                                    #
# ------------------------------------------------------------------ #
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/finetune_${TIMESTAMP}.log"

echo "로그 파일: ${LOG_FILE}"
echo ""

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port=29500 \
    "${SCRIPT_DIR}/finetune_canary_qwen.py" \
        --config-path="${SCRIPT_DIR}" \
        --config-name=canary_qwen_finetune \
        trainer.devices="${NUM_GPUS}" \
        trainer.num_nodes=1 \
        model.pretrained_weights=False \
        model.canary_qwen_pretrained_path="${NEMO_CKPT}" \
        "data.train_ds.input_cfg[0].input_cfg=${TRAIN_INPUT_CFG}" \
        "data.validation_ds.datasets.my_devset.input_cfg[0].cuts_path=${VAL_CUTS}" \
        exp_manager.explicit_log_dir="${RESULTS_DIR}" \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "=== 완료 ==="
echo "결과: ${RESULTS_DIR}"
echo "로그: ${LOG_FILE}"
