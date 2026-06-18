#!/usr/bin/env bash
# Shared pretraining launcher. Sourced by the per-arm 01_pretrain_*.sh scripts.
set -euo pipefail
COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$COMMON_DIR/env.sh"

run_pretrain() {
  local CONFIG_NAME="$1" ARM="$2"; shift 2
  _require_files NEMO_ROOT LIBRIHEAVY_TRAIN_MANIFEST SSL_DEV_MANIFEST
  local OUT="$EXP_ROOT/ssl/$ARM"
  mkdir -p "$OUT"
  echo ">>> SSL pretrain arm=$ARM config=$CONFIG_NAME -> $OUT"
  cd "$NEMO_ROOT"
  set -x
  python examples/asr/speech_pretraining/masked_token_pred_pretrain_streaming.py \
    --config-name="$CONFIG_NAME" \
    model.train_ds.manifest_filepath="$LIBRIHEAVY_TRAIN_MANIFEST" \
    model.validation_ds.manifest_filepath="$SSL_DEV_MANIFEST" \
    trainer.devices="$DEVICES" \
    trainer.num_nodes="$NUM_NODES" \
    trainer.precision="$PRECISION" \
    trainer.max_steps="$SSL_MAX_STEPS" \
    exp_manager.exp_dir="$OUT" \
    exp_manager.name="$ARM" \
    "$@"
}
