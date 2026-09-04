#!/bin/bash
# Evaluate every saved even-epoch CISLNet checkpoint on the frozen test set,
# then write results_c13.md + figures. Intended as an afterok dependent of train.
#
# Example:
#   RUN=c13_e35 EPOCHS=35 TEST_EVERY=2 sbatch test_epoch_sweep.sh
#SBATCH --job-name=cn_te
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --exclude=ccc0387,ccc0388
#SBATCH -A kf4-ic
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-12:00:00
#SBATCH --chdir=/projects/illinois/eng/bioe/rxb/akshat/VSlite
#SBATCH --output=/u/akshahi2/logs/%x/%j.out
#SBATCH --error=/u/akshahi2/logs/%x/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akshahi2@illinois.edu

set -euo pipefail
mkdir -p "/u/akshahi2/logs/${SLURM_JOB_NAME:-cn_te}"
: "${VSLITE_ROOT:=/projects/illinois/eng/bioe/rxb/akshat/VSlite}"
# shellcheck source=../../_lib/slurm_resolve.sh
source "$VSLITE_ROOT/jobs/_lib/slurm_resolve.sh"
VSLITE_ROOT="$(slurm_resolve_include_root jobs/_lib/vslite_env.sh "${BASH_SOURCE[0]:-$0}")" || exit 1
# shellcheck source=../../_lib/vslite_env.sh
source "$VSLITE_ROOT/jobs/_lib/vslite_env.sh"

NUM_CHANNELS="${NUM_CHANNELS:-13}"
RUN="${RUN:-c${NUM_CHANNELS}_e35}"
NAME="${NAME:-$RUN}"
RESULTS_DIR="${RESULTS_DIR:-$VSLITE_ROOT/results/cislnet}"
NUM_TEST="${NUM_TEST:-75000}"
EPOCHS="${EPOCHS:-35}"
TEST_EVERY="${TEST_EVERY:-2}"
# Canonical frozen eval is test.txt only (former val samples were merged in).
TEST_INCLUDES_VAL="${TEST_INCLUDES_VAL:-0}"

IR_INDICES="${IR_CHANNEL_INDICES:-}"
case "$NUM_CHANNELS" in
  3)  IR_INDICES="${IR_INDICES:-0,8,9}"; NUM_IR=3 ;;
  6)  NUM_IR=6 ;;
  10) NUM_IR=10 ;;
  13) NUM_IR=10 ;;
  *)  echo "Unsupported NUM_CHANNELS=$NUM_CHANNELS"; exit 1 ;;
esac

EXTRA=()
[[ -n "$IR_INDICES" ]] && EXTRA+=(--ir_channel_indices "$IR_INDICES")
[[ "${SAVE_TRIPLETS:-0}" == "1" ]] && EXTRA+=(--save_triplets)
if [[ -n "${TEST_LIST_PATH:-}" ]]; then EXTRA+=(--test_list_path "$TEST_LIST_PATH"); fi
if [[ -n "${VAL_LIST_PATH:-}" ]]; then EXTRA+=(--val_list_path "$VAL_LIST_PATH"); fi
EXTRA+=(--test_includes_val "$TEST_INCLUDES_VAL")

_epoch_list() {
  local e
  for ((e = TEST_EVERY; e <= EPOCHS; e += TEST_EVERY)); do
    echo "$e"
  done
  if (( EPOCHS % TEST_EVERY != 0 )); then
    echo "$EPOCHS"
  fi
}

mapfile -t EPOCH_LIST < <(_epoch_list)
echo "epoch sweep: RUN=$NAME EPOCHS=$EPOCHS TEST_EVERY=$TEST_EVERY epochs=${EPOCH_LIST[*]}"

n_ok=0
n_skip=0
n_miss=0
for which_epoch in "${EPOCH_LIST[@]}"; do
  ckpt_path="$RESULTS_DIR/$NAME/${which_epoch}_net_G.pth"
  metrics_path="$RESULTS_DIR/test/${NAME}_${which_epoch}/patch_eval_test/metrics.json"
  if [[ -f "$metrics_path" ]]; then
    echo "epoch $which_epoch -> skip (metrics exist)"
    n_skip=$((n_skip + 1))
    continue
  fi
  if [[ ! -f "$ckpt_path" ]]; then
    echo "epoch $which_epoch -> skip (missing ckpt $ckpt_path)" >&2
    n_miss=$((n_miss + 1))
    continue
  fi
  echo "epoch $which_epoch -> eval $ckpt_path"
  python "$VSLITE_ROOT/scripts/vs_infer_and_metrics.py" \
    --name "$NAME" \
    --checkpoints_dir "$RESULTS_DIR" \
    --dataset_mode VS \
    --model contrastive_pix2pix \
    --dataroot "$DATAROOT" \
    --input_nc "$NUM_CHANNELS" \
    --num_ir_channels "$NUM_IR" \
    --which_epoch "$which_epoch" \
    --results_dir "$RESULTS_DIR" \
    --gpu_ids 0 \
    --num_test "$NUM_TEST" \
    --fid_max_items "${VSLITE_FID_MAX_ITEMS:-$NUM_TEST}" \
    --fid_batch_size "${VSLITE_FID_BATCH_SIZE:-64}" \
    "${EXTRA[@]}"
  n_ok=$((n_ok + 1))
done

echo "epoch sweep done: evaluated=$n_ok skipped_existing=$n_skip missing_ckpt=$n_miss"

python "$VSLITE_ROOT/scripts/plot_c13_epoch_eval.py" \
  --run "$NAME" \
  --train-patches "${TRAIN_PATCHES:-60047}" \
  || echo "WARNING: plot_c13_epoch_eval.py failed (metrics were still written)" >&2

if (( n_ok + n_skip == 0 )); then
  echo "ERROR: no checkpoints evaluated" >&2
  exit 1
fi
