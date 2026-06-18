#!/usr/bin/env bash
# End-to-end driver for the 3 core arms (A vs B0 vs B*): pretrain -> finetune -> eval, in order.
# Long-running. For real use you'll likely launch the pretrains separately on a cluster; this is a
# convenience / smoke-test driver. Set ARMS to a subset to run fewer. Add FSQ_streaming to include FSQ.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/env.sh"

ARMS=("A_offline" "B0_streaming" "Bstar_dualmode")
declare -A PRETRAIN=(
  [A_offline]="$DIR/01_pretrain_A_offline.sh"
  [B0_streaming]="$DIR/01_pretrain_B0_streaming.sh"
  [Bstar_dualmode]="$DIR/01_pretrain_Bstar_dualmode.sh"
  [FSQ_streaming]="$DIR/01_pretrain_FSQ_streaming.sh"
)

for ARM in "${ARMS[@]}"; do
  echo "######## [$ARM] PRETRAIN ########"; bash "${PRETRAIN[$ARM]}"
  echo "######## [$ARM] FINETUNE ########"; bash "$DIR/02_finetune.sh" "$ARM"
  echo "######## [$ARM] EVAL SWEEP ######"; bash "$DIR/03_eval_sweep.sh" "$ARM"
done
echo "ALL ARMS DONE. Compare WER per chunk size across arms (expect B* <= B0 <= A; biggest gain at small chunks)."
