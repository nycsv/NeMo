#!/usr/bin/env bash
# ============================================================================
# EDIT THIS FILE ONLY (the data + compute part). All run scripts source it.
# ============================================================================
# Streaming BEST-RQ SSL experiment: A vs B0 vs B-proposed (dual-mode), optional FSQ.
# Pretrain on Libriheavy, finetune on train-clean-100, evaluate across chunk sizes.

# --- repo root (absolute path to the NeMo tree) ---------------------------
export NEMO_ROOT="/home/eesung/work/nemotron-multilingual-asr/NeMo-main"

# --- DATA: SSL pre-training (Libriheavy) ----------------------------------
# Transcripts are ignored by the SSL objective; only audio is used.
export LIBRIHEAVY_TRAIN_MANIFEST="???"   # e.g. /data/libriheavy/libriheavy_cuts_large.jsonl  (or NeMo manifest)
export SSL_DEV_MANIFEST="???"            # small dev manifest for val_loss / checkpointing

# --- DATA: ASR fine-tuning (train-clean-100) ------------------------------
export FT_TRAIN_MANIFEST="???"           # e.g. /data/LibriSpeech/train_clean_100.json
export FT_DEV_MANIFEST="???"             # e.g. /data/LibriSpeech/dev_clean.json
export TOKENIZER_DIR="???"               # BPE tokenizer dir (build once, e.g. process_asr_text_tokenizer.py)
export TOKENIZER_TYPE="bpe"

# --- DATA: evaluation -----------------------------------------------------
export TEST_MANIFEST="???"               # e.g. /data/LibriSpeech/test_clean.json

# --- OUTPUT ---------------------------------------------------------------
export EXP_ROOT="${NEMO_ROOT}/exp/streaming_ssl"   # all checkpoints/logs land here

# --- COMPUTE --------------------------------------------------------------
export DEVICES="-1"          # -1 = all visible GPUs
export NUM_NODES="1"
export PRECISION="bf16"      # 16 / 32 / bf16 (config default is 32)
export SSL_MAX_STEPS="500000"
# train-clean-100 is small -> shorter finetune by default
export FT_MAX_STEPS="50000"
export FT_WARMUP="5000"

# --- chunk-size sweep for evaluation (att_context_size = [left,right]) -----
# right context -> look-ahead at subsampling 8: 13~1.04s, 6~0.48s, 1~0.08s, 0=0s
export EVAL_CTXS=("[70,13]" "[70,6]" "[70,1]" "[70,0]")

# ============================================================================
# Helpers (do not edit below) ------------------------------------------------
# ============================================================================
_require_files() {
  local missing=0
  for v in "$@"; do
    if [[ "${!v}" == "???" || -z "${!v}" ]]; then
      echo "ERROR: \$$v is not set in env.sh (currently '${!v}')." >&2
      missing=1
    fi
  done
  [[ $missing -eq 0 ]] || { echo "Fill in the data paths in env.sh and re-run." >&2; exit 1; }
}

# find the newest .nemo under a directory (used to locate produced checkpoints)
latest_nemo() { find "$1" -name '*.nemo' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-; }
