#!/bin/bash
# Submit c10: 10-channel CISLNet on the default c13 architecture + loss
# (cbam / resnet50 / standard decoder / srgb; NCE 10 + GAN 1.0, L1/L2/VGG off).
# This is the renamed successor of the legacy 15-epoch vs10 baseline
# (FRAMEWORK run-tag convention: variant c + channels = c10).
#
# Chain (login driver, calls sbatch):
#   train hop1 (EPOCHS, SAVE_EPOCH_FREQ=5, epoch-end LPIPS diagnostics)
#     → train hop2 (afterany, CONTINUE_TRAIN=true, resumes from iter.txt)
#       → te epoch sweep  (afterok: test/val/train LPIPS+FID+PSNR at every
#                          saved epoch + results_figures_c10 report)
#       → te tracking row (afterok: latest ckpt → experiments/tracking.csv
#                          upload, same as the c13 ablation experiments)
#       → wsi + remote upload (afterok: AUTO_BEST_LPIPS epoch → DZI → viewer
#                          project campus_ablations_bestlpips)
#
# Usage:
#   ./jobs/cislnet/experiments/submit_c10.sh
#   DRY_RUN=1 ./jobs/cislnet/experiments/submit_c10.sh
#   EPOCHS=35 SAVE_EPOCH_FREQ=2 ./jobs/cislnet/experiments/submit_c10.sh
#   CONTINUE_TRAIN=true ./jobs/cislnet/experiments/submit_c10.sh   # resume if c10 exists

set -euo pipefail
: "${VSLITE_ROOT:=/projects/illinois/eng/bioe/rxb/akshat/VSlite}"
# shellcheck source=_lib/vslite_env.sh
source "$VSLITE_ROOT/jobs/_lib/vslite_env.sh"
# shellcheck source=_lib/slurm_defaults.sh
source "$VSLITE_ROOT/jobs/_lib/slurm_defaults.sh"

NUM_CHANNELS="${NUM_CHANNELS:-10}"
RUN="${RUN:-c10}"
NAME="${NAME:-$RUN}"
EPOCHS="${EPOCHS:-30}"
SAVE_EPOCH_FREQ="${SAVE_EPOCH_FREQ:-5}"
TEST_EVERY="${TEST_EVERY:-$SAVE_EPOCH_FREQ}"
NUM_TEST="${NUM_TEST:-75000}"
DRY_RUN="${DRY_RUN:-0}"
CONTINUE_TRAIN="${CONTINUE_TRAIN:-false}"

TRAIN_TIME="${TRAIN_TIME:-3-00:00:00}"
TRAIN_TIME2="${TRAIN_TIME2:-$TRAIN_TIME}"
SWEEP_TIME="${SWEEP_TIME:-1-12:00:00}"
TRACK_TIME="${TRACK_TIME:-06:00:00}"
WSI_TIME="${WSI_TIME:-24:00:00}"
EXCLUDE="${SLURM_EXCLUDE_NODES}"
LOG_ROOT="${LOG_ROOT:-/u/akshahi2/logs}"
TRAIN_PATCHES="${TRAIN_PATCHES:-60047}"

# Remote viewer project (same as the c13 best-LPIPS WSI uploads).
PROJECT="${PROJECT:-campus_ablations_bestlpips}"
PROJECT_DISPLAY="${PROJECT_DISPLAY_NAME:-Campus_ablations_best_LPIPS_WSI}"
WSI_FOLDERS="${WSI_FOLDERS:-10442-1:10442-2:10445-2:10445-3:10445-4:3840-1:3840-2:3840-3:3840-4:3840-5}"

JOBS_DIR="$VSLITE_ROOT/jobs/cislnet"
TRAIN_SH="$JOBS_DIR/train/train.sh"
SWEEP_SH="$JOBS_DIR/test/test_epoch_sweep.sh"
TRACK_SH="$JOBS_DIR/experiments/eval_only.sh"
WSI_SH="$JOBS_DIR/wsi/wsi.sh"

for d in cn_tr_${NAME} cn_te_${NAME} cn_te_${NAME}_trk cn_wsi_${NAME}; do
  mkdir -p "$LOG_ROOT/$d"
done

_submit_parsable() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "dryrun_${RANDOM}"
    return 0
  fi
  sbatch --parsable "$@"
}

echo "c10 submit: RUN=$NAME NUM_CHANNELS=$NUM_CHANNELS EPOCHS=$EPOCHS SAVE_EPOCH_FREQ=$SAVE_EPOCH_FREQ TEST_EVERY=$TEST_EVERY CONTINUE_TRAIN=$CONTINUE_TRAIN exclude=$EXCLUDE"

export_vars="ALL,NUM_CHANNELS=$NUM_CHANNELS,EPOCHS=$EPOCHS,SAVE_EPOCH_FREQ=$SAVE_EPOCH_FREQ"
export_vars+=",RUN=$NAME,NAME=$NAME,CONTINUE_TRAIN=$CONTINUE_TRAIN"
J_TRAIN1=$(_submit_parsable --job-name="cn_tr_${NAME}" --time="$TRAIN_TIME" --exclude="$EXCLUDE" \
  --export="$export_vars" "$TRAIN_SH")
echo "train $NAME hop1 -> job $J_TRAIN1 (time=$TRAIN_TIME save_epoch_freq=$SAVE_EPOCH_FREQ)"

export_vars2="${export_vars/,CONTINUE_TRAIN=$CONTINUE_TRAIN/,CONTINUE_TRAIN=true}"
J_TRAIN2=$(_submit_parsable --job-name="cn_tr_${NAME}" --time="$TRAIN_TIME2" --exclude="$EXCLUDE" \
  --dependency="afterany:${J_TRAIN1}" --export="$export_vars2" "$TRAIN_SH")
echo "train $NAME hop2 -> job $J_TRAIN2 (afterany:$J_TRAIN1 CONTINUE_TRAIN=true)"

sweep_export="ALL,NUM_CHANNELS=$NUM_CHANNELS,EPOCHS=$EPOCHS,TEST_EVERY=$TEST_EVERY"
sweep_export+=",RUN=$NAME,NAME=$NAME,NUM_TEST=$NUM_TEST,TEST_INCLUDES_VAL=0"
sweep_export+=",TRAIN_PATCHES=$TRAIN_PATCHES,PLOT_OUT_DIR=$VSLITE_ROOT/results_figures_c10"
J_SWEEP=$(_submit_parsable --job-name="cn_te_${NAME}" --time="$SWEEP_TIME" --exclude="$EXCLUDE" \
  --dependency="afterok:${J_TRAIN2}" --export="$sweep_export" "$SWEEP_SH")
echo "te sweep $NAME every ${TEST_EVERY} ep -> job $J_SWEEP (afterok:$J_TRAIN2 time=$SWEEP_TIME)"

track_export="ALL,NUM_CHANNELS=$NUM_CHANNELS,NUM_IR=10,IR_INDICES="
track_export+=",NAME=$NAME,WHICH_EPOCH=latest,TRACK_RUN=$NAME"
track_export+=",TRACK_GROUP=${TRACK_GROUP:-EXP}"
track_export+=",TRACK_NOTES=${TRACK_NOTES:-c10_default_c13_arch_ep${EPOCHS}_save${SAVE_EPOCH_FREQ}}"
track_export+=",RESULTS_NAME=$NAME,NUM_TEST=$NUM_TEST"
J_TRACK=$(_submit_parsable --job-name="cn_te_${NAME}_trk" --time="$TRACK_TIME" --exclude="$EXCLUDE" \
  --dependency="afterok:${J_TRAIN2}" --export="$track_export" "$TRACK_SH")
echo "te tracking $NAME (latest -> experiments/tracking.csv) -> job $J_TRACK (afterok:$J_TRAIN2)"

wsi_export="ALL,NUM_CHANNELS=$NUM_CHANNELS,RUN=$NAME,NAME=$NAME"
wsi_export+=",AUTO_BEST_LPIPS=1,OUT_TAG=bestlpips,REMOTE_UPLOAD=1"
wsi_export+=",PROJECT=$PROJECT,PROJECT_DISPLAY_NAME=$PROJECT_DISPLAY"
wsi_export+=",OUTPUT_TYPE=${NAME}-bestlpips,OVERWRITE=1"
wsi_export+=",WSI_FOLDERS=$WSI_FOLDERS,WSI_IMAGE_FORMAT=jpeg,WSI_JPEG_QUALITY=92"
J_WSI=$(_submit_parsable --job-name="cn_wsi_${NAME}" --time="$WSI_TIME" --exclude="$EXCLUDE" \
  --dependency="afterok:${J_TRAIN2}" --export="$wsi_export" "$WSI_SH")
echo "wsi+upload $NAME (AUTO_BEST_LPIPS -> $PROJECT) -> job $J_WSI (afterok:$J_TRAIN2)"

echo "Done. ckpts=$VSLITE_ROOT/results/cislnet/$NAME/ sweep-report=results_figures_c10/ tracking=results/cislnet/experiments/tracking.csv"
