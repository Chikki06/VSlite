#!/bin/bash
# Queue VIS-only + kernel-width ablations: train(+patch test) then WSI remote
# upload into the same viewer project as the campus_ablations_bestlpips runs.
#
# Chain per run (mirrors submit_abc.sh + submit_best_lpips_ablations.sh):
#   train hop1 -> train hop2 (afterany, CONTINUE_TRAIN=true) -> WSI+upload (afterok)
# train_eval.sh runs train + patch eval (75k, FID/PSNR/SSIM); the WSI job uses
# AUTO_BEST_LPIPS and REMOTE_UPLOAD=1 with OUTPUT_TYPE=<run>-bestlpips as the
# modality name in PROJECT=campus_ablations_bestlpips.
#
# Usage:
#   bash jobs/cislnet/experiments/submit_vis_kernel.sh
#   DRY_RUN=1 bash jobs/cislnet/experiments/submit_vis_kernel.sh
#   EPOCHS=20 TRAIN_CHAIN=1 bash jobs/cislnet/experiments/submit_vis_kernel.sh
set -euo pipefail
ROOT="/projects/illinois/eng/bioe/rxb/akshat/VSlite"
EPOCHS="${EPOCHS:-20}"
TRAIN_CHAIN="${TRAIN_CHAIN:-1}"
DRY_RUN="${DRY_RUN:-0}"

PROJECT="${CAMPUS_PROJECT:-campus_ablations_bestlpips}"
# NOTE: plain ${PROJECT:-...} is unsafe here — login shells on this machine
# export PROJECT=<VSLite root>, which would silently misroute uploads.
PROJECT_DISPLAY="${PROJECT_DISPLAY_NAME:-Campus_ablations_best_LPIPS_WSI}"
WSI_PARTITION="${WSI_PARTITION:-IllinoisComputes-GPU}"
WSI_GRES="${WSI_GRES:-gpu:A100:1}"
WSI_TIME="${WSI_TIME:-24:00:00}"
WSI_EXCLUDE="${WSI_EXCLUDE:-ccc0388}"
WSI_SH="$ROOT/jobs/cislnet/wsi/wsi.sh"

# script|extra-train/wsi-env (channels etc.)
CONFIGS=(
  "visonly_c13.sh|VIS_ONLY=1"
  "dec25_c13.sh|"
  "dec10_c13.sh|"
  "dec75_c13.sh|"
  "encdec50_c13.sh|"
)

mkdir -p /u/akshahi2/logs/cn_visonly /u/akshahi2/logs/cn_dec25 /u/akshahi2/logs/cn_dec10 \
  /u/akshahi2/logs/cn_dec75 /u/akshahi2/logs/cn_encdec50 /u/akshahi2/logs/cn_wsi_best

_submit() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY sbatch $*" >&2
    echo "dryrun_${RANDOM}"
    return 0
  fi
  sbatch --parsable "$@"
}

echo "root=$ROOT EPOCHS=$EPOCHS TRAIN_CHAIN=$TRAIN_CHAIN project=$PROJECT"

for spec in "${CONFIGS[@]}"; do
  IFS='|' read -r script extra <<<"$spec"
  name="${script%.sh}"
  job="cn_tr_${name#*_}"
  # Hop 1: fresh train (+patch test inside train_eval.sh).
  export1="ALL,EPOCHS=${EPOCHS}"
  [[ -n "$extra" ]] && export1+=",${extra}"
  j1=$(_submit \
    --job-name="$job" \
    --export="$export1" \
    "$ROOT/jobs/cislnet/experiments/${script}")
  echo "train $name hop1 -> $j1"

  # Hop 2: resume-to-finish on any exit (walltime-safe), then patch test.
  export2="${export1},CONTINUE_TRAIN=true"
  if [[ "$TRAIN_CHAIN" == "1" ]]; then
    j2=$(_submit \
      --job-name="$job" \
      --dependency="afterany:${j1}" \
      --export="$export2" \
      "$ROOT/jobs/cislnet/experiments/${script}")
    echo "train $name hop2 -> $j2 (afterany:$j1 CONTINUE_TRAIN=true)"
  else
    j2="$j1"
  fi

  # WSI infer->DZI->upload at best-LPIPS epoch, after training succeeds.
  # OUTPUT_TYPE doubles as the modality name in the viewer project.
  OUTPUT_TYPE="${name}-bestlpips"
  export_wsi="ALL,NAME=${name},RUN=${name},AUTO_BEST_LPIPS=1,OUT_TAG=bestlpips,REMOTE_UPLOAD=1,PROJECT=${PROJECT},PROJECT_DISPLAY_NAME=${PROJECT_DISPLAY},OUTPUT_TYPE=${OUTPUT_TYPE},OVERWRITE=1"
  [[ -n "$extra" ]] && export_wsi+=",${extra}"
  jw=$(_submit \
    --partition="$WSI_PARTITION" \
    --gres="$WSI_GRES" \
    --time="$WSI_TIME" \
    --exclude="$WSI_EXCLUDE" \
    --job-name="cn_wsi_${name}" \
    --dependency="afterok:${j2}" \
    --output="/u/akshahi2/logs/cn_wsi_best/%j.out" \
    --error="/u/akshahi2/logs/cn_wsi_best/%j.err" \
    --export="$export_wsi" \
    "$WSI_SH")
  echo "wsi   $name upload -> $jw (afterok:$j2 OUTPUT_TYPE=$OUTPUT_TYPE)"
done

echo "Done. Viewer project: $PROJECT"
