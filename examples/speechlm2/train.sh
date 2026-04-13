#!/bin/bash
# =============================================================================
#  run_finetune.sh
#
#  Single-node multi-GPU fine-tuning for nvidia/canary-qwen-2.5b
#  Strategy : FSDP (Fully Sharded Data Parallel)
#  Launcher  : torchrun  (no SLURM / srun required)
#
#  Usage:
#    chmod +x run_finetune.sh
#    ./run_finetune.sh
#
#  Background:
#    nohup ./run_finetune.sh > logs/nohup.out 2>&1 &
# =============================================================================

# ------------------------------------------------------------------ #
#  User settings — edit only this section                              #
# ------------------------------------------------------------------ #

# GPU IDs to use  (e.g. "0,1,2,3"  or  "0,1")
CUDA_VISIBLE="0,1,2,3"

# Number of GPUs  (must match the count above)
NUM_GPUS=4

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEMO_CKPT="${SCRIPT_DIR}/canary-qwen-2.5b.nemo"
TRAIN_INPUT_CFG="${SCRIPT_DIR}/input_cfg.yaml"
VAL_CUTS="/path/to/val_cuts.jsonl.gz"
RESULTS_DIR="${SCRIPT_DIR}/results"
LOG_DIR="${SCRIPT_DIR}/logs"

# ------------------------------------------------------------------ #
#  Pre-flight checks                                                   #
# ------------------------------------------------------------------ #
set -e  # exit immediately on error

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

echo "=================================================="
echo "  canary-qwen-2.5b  |  FSDP Fine-tuning"
echo "  GPUs   : ${CUDA_VISIBLE}  (${NUM_GPUS} devices)"
echo "  CKPT   : ${NEMO_CKPT}"
echo "  Output : ${RESULTS_DIR}"
echo "=================================================="

if [ ! -f "${NEMO_CKPT}" ]; then
    echo "[ERROR] Checkpoint not found: ${NEMO_CKPT}"
    echo "        Download it first via save_modules.py or HuggingFace Hub."
    exit 1
fi

# ------------------------------------------------------------------ #
#  Environment variables                                               #
# ------------------------------------------------------------------ #
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE}"

# --- NCCL (single-node) ---
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1           # disable InfiniBand (not present on single node)
export NCCL_P2P_DISABLE=0          # enable NVLink P2P
export NCCL_SOCKET_IFNAME=lo       # use loopback for single-node comms

# --- FSDP / PyTorch ---
export TORCH_NCCL_BLOCKING_WAIT=1
export OMP_NUM_THREADS=4

# FSDP requires this to avoid timeout on large model sharding at startup
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Increase timeout for FSDP all-gather on large models (seconds)
export NCCL_TIMEOUT=1800

# ------------------------------------------------------------------ #
#  Launch with torchrun                                                #
# ------------------------------------------------------------------ #
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/finetune_fsdp_${TIMESTAMP}.log"

echo "Log file : ${LOG_FILE}"
echo ""

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NUM_GPUS}" \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    "${SCRIPT_DIR}/finetune_canary_qwen.py" \
        --config-path="${SCRIPT_DIR}" \
        --config-name=canary_qwen_finetune \
        trainer.devices="${NUM_GPUS}" \
        trainer.num_nodes=1 \
        trainer.precision=bf16-true \
        model.pretrained_weights=False \
        model.canary_qwen_pretrained_path="${NEMO_CKPT}" \
        "data.train_ds.input_cfg[0].input_cfg=${TRAIN_INPUT_CFG}" \
        "data.validation_ds.datasets.my_devset.input_cfg[0].cuts_path=${VAL_CUTS}" \
        exp_manager.explicit_log_dir="${RESULTS_DIR}" \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "=== Done ==="
echo "Results : ${RESULTS_DIR}"
echo "Log     : ${LOG_FILE}"
