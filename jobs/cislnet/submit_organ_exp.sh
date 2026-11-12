#!/bin/bash
# Submit organ-agnostic vs organ-specific CISLNet c13 trains + dependent FID/LPIPS tests.
# Login-node driver only (calls sbatch). No GPU needed here.
#
# Compute budget is matched *iterations* (samples seen), not local epochs.
# All-samples × BASE_EPOCHS (default 30) is the baseline; smaller pools train
# more local epochs so every run sees the same number of samples.
#
# Usage:
#   ./jobs/cislnet/submit_organ_exp.sh
#   EPOCHS=30 DRY_RUN=1 ./jobs/cislnet/submit_organ_exp.sh
#   BUILD_LISTS=0 ./jobs/cislnet/submit_organ_exp.sh   # skip list rebuild
#
# Resume incomplete trains + requeue holdout tests (default CONTINUE_TRAIN=true):
#   BUILD_LISTS=0 CONTINUE_TRAIN=true \
#     TRAIN_TIME=3-00:00:00 ./jobs/cislnet/submit_organ_exp.sh
#
# Release tests for checkpoints that appeared mid-train (no new trains):
#   BUILD_LISTS=0 SKIP_TRAIN=1 TEST_EQ_EPOCHS=15,20,30 ./jobs/cislnet/submit_organ_exp.sh
#
# Each train is submitted twice: hop 2 is --dependency=afterany:<hop1> with
# CONTINUE_TRAIN=true so a 3-day walltime timeout still resumes. Holdout tests
# depend on hop 2 with afterok.
#
# If ${epoch}_net_G.pth already exists, that epoch's tests still wait on hop 2
# when a train is submitted (avoids racing a resume). Already-finished
# metrics.json with lpips_convnext_tcga_mean are skipped.

set -euo pipefail
: "${VSLITE_ROOT:=/projects/illinois/eng/bioe/rxb/akshat/VSlite}"
# shellcheck source=_lib/slurm_resolve.sh
source "$VSLITE_ROOT/jobs/_lib/slurm_resolve.sh"
VSLITE_ROOT="$(slurm_resolve_include_root jobs/_lib/vslite_env.sh "${BASH_SOURCE[0]:-$0}")" || exit 1
# shellcheck source=_lib/vslite_env.sh
source "$VSLITE_ROOT/jobs/_lib/vslite_env.sh"
# shellcheck source=_lib/slurm_defaults.sh
source "$VSLITE_ROOT/jobs/_lib/slurm_defaults.sh"

NUM_CHANNELS="${NUM_CHANNELS:-13}"
# BASE_EPOCHS is the all-samples epoch budget; per-run local epochs are derived.
EPOCHS="${EPOCHS:-30}"
NUM_TEST="${NUM_TEST:-75000}"
DRY_RUN="${DRY_RUN:-0}"
BUILD_LISTS="${BUILD_LISTS:-1}"
EXCLUDE="${SLURM_EXCLUDE_NODES}"
CONTINUE_TRAIN="${CONTINUE_TRAIN:-true}"
# Comma-separated run names to train; empty = all five organ-exp runs.
# SKIP_TRAIN=1 skips every train (tests only / release ready ckpts).
TRAIN_RUNS="${TRAIN_RUNS:-}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
# Comma-separated run names to test; empty = all five. Use with TRAIN_RUNS to
# avoid duplicating tests for a run that is still training from an earlier submit.
TEST_RUNS="${TEST_RUNS:-}"
# All-samples-equivalent checkpoint grid (not local epochs).
TEST_EQ_EPOCHS="${TEST_EQ_EPOCHS:-${TEST_EPOCHS:-5,10,15,20,25,30}}"
# Chain a second hop so walltime timeout can resume (afterany).
TRAIN_CHAIN="${TRAIN_CHAIN:-1}"
# Optional sbatch --time override for train jobs (e.g. 3-00:00:00).
TRAIN_TIME="${TRAIN_TIME:-${SLURM_TIME_TRAIN:-3-00:00:00}}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-$VSLITE_ROOT/results/cislnet}"
RESULTS_DIR="${RESULTS_DIR:-$CHECKPOINTS_DIR}"
LOG_ROOT="${LOG_ROOT:-/u/akshahi2/logs}"
BATCH_SIZE="${BATCH_SIZE:-10}"

ORG_DIR="${ORG_DIR:-$VSLITE_ROOT/dataset/organ_exp}"
JOBS_DIR="$VSLITE_ROOT/jobs/cislnet"
TRAIN_SH="$JOBS_DIR/train/train.sh"
TEST_SH="$JOBS_DIR/test/test.sh"
SCHED_PY="$VSLITE_ROOT/scripts/organ_exp_schedule.py"
MANIFEST="$ORG_DIR/manifest.json"

_submit_parsable() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "dryrun_${RANDOM}"
    return 0
  fi
  sbatch --parsable "$@"
}

_should_train() {
  local run="$1"
  if [[ "$SKIP_TRAIN" == "1" ]]; then
    return 1
  fi
  if [[ -z "$TRAIN_RUNS" ]]; then
    return 0
  fi
  local item
  IFS=',' read -r -a _runs <<< "$TRAIN_RUNS"
  for item in "${_runs[@]}"; do
    if [[ "$item" == "$run" ]]; then
      return 0
    fi
  done
  return 1
}

_should_test() {
  local run="$1"
  if [[ -z "$TEST_RUNS" ]]; then
    return 0
  fi
  local item
  IFS=',' read -r -a _truns <<< "$TEST_RUNS"
  for item in "${_truns[@]}"; do
    if [[ "$item" == "$run" ]]; then
      return 0
    fi
  done
  return 1
}

_metrics_has_lpips() {
  local path="$1"
  python - "$path" <<'PY'
import json, sys
p = sys.argv[1]
try:
    m = json.load(open(p))
except Exception:
    raise SystemExit(1)
v = m.get("lpips_convnext_tcga_mean")
raise SystemExit(0 if v is not None else 1)
PY
}

if [[ "$BUILD_LISTS" == "1" ]]; then
  python "$VSLITE_ROOT/scripts/build_organ_exp_lists.py" --vslite-root "$VSLITE_ROOT" --out-dir "$ORG_DIR"
fi

for f in \
  holdout_prostate.txt holdout_kidney.txt holdout_uterus.txt holdout_all.txt \
  train_oa_all.txt train_oa_top3.txt \
  train_os_prostate.txt train_os_kidney.txt train_os_uterus.txt \
  val_oa_all.txt val_oa_top3.txt \
  val_os_prostate.txt val_os_kidney.txt val_os_uterus.txt \
  manifest.json
do
  if [[ ! -f "$ORG_DIR/$f" ]]; then
    echo "Missing $ORG_DIR/$f — run build_organ_exp_lists.py first" >&2
    exit 1
  fi
done

# shellcheck disable=SC1090
eval "$(python "$SCHED_PY" --manifest "$MANIFEST" --base-epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" --test-eq-epochs "$TEST_EQ_EPOCHS" --export-bash)"

python "$SCHED_PY" --manifest "$MANIFEST" --base-epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" --test-eq-epochs "$TEST_EQ_EPOCHS"
python "$SCHED_PY" --manifest "$MANIFEST" --base-epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" --test-eq-epochs "$TEST_EQ_EPOCHS" --json \
  > "$ORG_DIR/iter_schedule.json"

echo "Organ exp submit: NUM_CHANNELS=$NUM_CHANNELS BASE_EPOCHS=$EPOCHS target_samples=${ORG_TARGET_SAMPLES} TEST_EQ_EPOCHS=$TEST_EQ_EPOCHS CONTINUE_TRAIN=$CONTINUE_TRAIN SKIP_TRAIN=$SKIP_TRAIN TRAIN_RUNS=${TRAIN_RUNS:-all} TEST_RUNS=${TEST_RUNS:-all} TRAIN_CHAIN=$TRAIN_CHAIN TRAIN_TIME=$TRAIN_TIME ORG_DIR=$ORG_DIR exclude=$EXCLUDE"

# ---- trains ----
# Holdouts are TEST only. Val lists are carved from train pools — never reuse
# holdout as --val_list_path (VS_dataset excludes test IDs from val → empty set).
declare -A J_TRAIN

_submit_train() {
  local run="$1"
  local train_list="$2"
  local test_list="$3"
  local val_list="$4"
  if ! _should_train "$run"; then
    if [[ "$SKIP_TRAIN" == "1" ]]; then
      echo "train $run -> skipped (SKIP_TRAIN=1)"
    else
      echo "train $run -> skipped (not in TRAIN_RUNS)"
    fi
    return 0
  fi
  local local_epochs="${ORG_EPOCHS[$run]}"
  local save_epochs="${ORG_SAVE_EPOCHS[$run]}"
  if [[ -z "$local_epochs" ]]; then
    echo "No schedule for $run" >&2
    exit 1
  fi
  local job_name="cn_tr_${run}"
  mkdir -p "$LOG_ROOT/$job_name"
  local export_vars
  export_vars="ALL,NUM_CHANNELS=$NUM_CHANNELS,EPOCHS=$local_epochs,SAVE_EPOCHS=$save_epochs"
  export_vars+=",BATCH_SIZE=$BATCH_SIZE,RUN=$run,NAME=$run"
  export_vars+=",TRAIN_LIST_PATH=$train_list,TEST_LIST_PATH=$test_list,VAL_LIST_PATH=$val_list"
  export_vars+=",CONTINUE_TRAIN=$CONTINUE_TRAIN"
  local sbatch_extra=()
  if [[ -n "$TRAIN_TIME" ]]; then
    sbatch_extra+=(--time="$TRAIN_TIME")
  fi
  local jid
  jid=$(_submit_parsable "${sbatch_extra[@]}" --job-name="$job_name" --exclude="$EXCLUDE" --export="$export_vars" "$TRAIN_SH")
  echo "train $run hop1 -> job $jid (local_epochs=$local_epochs save_epochs=$save_epochs continue=$CONTINUE_TRAIN)"

  if [[ "$TRAIN_CHAIN" == "1" ]]; then
    local export_vars2="${export_vars/,CONTINUE_TRAIN=$CONTINUE_TRAIN/,CONTINUE_TRAIN=true}"
    if [[ "$export_vars2" != *"CONTINUE_TRAIN=true"* ]]; then
      export_vars2+=",CONTINUE_TRAIN=true"
    fi
    local jid2
    jid2=$(_submit_parsable "${sbatch_extra[@]}" --job-name="$job_name" --exclude="$EXCLUDE" \
      --dependency="afterany:${jid}" --export="$export_vars2" "$TRAIN_SH")
    echo "train $run hop2 -> job $jid2 (afterany:$jid CONTINUE_TRAIN=true)"
    J_TRAIN[$run]="$jid2"
  else
    J_TRAIN[$run]="$jid"
  fi
}

_submit_train c13_oa_all \
  "$ORG_DIR/train_oa_all.txt" \
  "$ORG_DIR/holdout_all.txt" \
  "$ORG_DIR/val_oa_all.txt"

_submit_train c13_oa_top3 \
  "$ORG_DIR/train_oa_top3.txt" \
  "$ORG_DIR/holdout_all.txt" \
  "$ORG_DIR/val_oa_top3.txt"

_submit_train c13_os_prostate \
  "$ORG_DIR/train_os_prostate.txt" \
  "$ORG_DIR/holdout_prostate.txt" \
  "$ORG_DIR/val_os_prostate.txt"

_submit_train c13_os_kidney \
  "$ORG_DIR/train_os_kidney.txt" \
  "$ORG_DIR/holdout_kidney.txt" \
  "$ORG_DIR/val_os_kidney.txt"

_submit_train c13_os_uterus \
  "$ORG_DIR/train_os_uterus.txt" \
  "$ORG_DIR/holdout_uterus.txt" \
  "$ORG_DIR/val_os_uterus.txt"

# ---- holdout tests: each (model, organ, local epoch matching the eq grid) ----
_submit_fid() {
  local ckpt_name="$1"
  local organ="$2"   # prostate|kidney|uterus
  local which_epoch="$3"
  if ! _should_test "$ckpt_name"; then
    echo "test  ${ckpt_name}__hold_${organ} epoch=$which_epoch -> skipped (not in TEST_RUNS)"
    return 0
  fi
  local results_name="${ckpt_name}__hold_${organ}"
  local holdout="$ORG_DIR/holdout_${organ}.txt"
  local ckpt_path="$CHECKPOINTS_DIR/$ckpt_name/${which_epoch}_net_G.pth"
  if [[ "$which_epoch" == "latest" ]]; then
    ckpt_path="$CHECKPOINTS_DIR/$ckpt_name/latest_net_G.pth"
  fi

  local metrics_path
  if [[ "$which_epoch" == "latest" ]]; then
    metrics_path="$RESULTS_DIR/test/${results_name}/patch_eval_test/metrics.json"
  else
    metrics_path="$RESULTS_DIR/test/${results_name}_${which_epoch}/patch_eval_test/metrics.json"
  fi
  if [[ -f "$metrics_path" ]] && _metrics_has_lpips "$metrics_path"; then
    echo "test  $results_name epoch=$which_epoch -> skipped (lpips metrics exist: $metrics_path)"
    return 0
  fi

  local dep_args=()
  local dep_note="no-dep"
  if [[ -n "${J_TRAIN[$ckpt_name]:-}" ]]; then
    dep_args=(--dependency="afterok:${J_TRAIN[$ckpt_name]}")
    dep_note="afterok:${J_TRAIN[$ckpt_name]}"
  elif [[ -f "$ckpt_path" ]]; then
    dep_note="ckpt-ready:$ckpt_path"
  else
    echo "test  $results_name epoch=$which_epoch -> skipped (no ckpt $ckpt_path and no train job)"
    return 0
  fi

  local job_name="cn_te_${results_name}_e${which_epoch}"
  mkdir -p "$LOG_ROOT/$job_name"
  local export_vars
  export_vars="ALL,NUM_CHANNELS=$NUM_CHANNELS,RUN=$ckpt_name,NAME=$ckpt_name"
  export_vars+=",RESULTS_NAME=$results_name"
  export_vars+=",TEST_LIST_PATH=$holdout,TEST_INCLUDES_VAL=0,NUM_TEST=$NUM_TEST"
  export_vars+=",WHICH_EPOCH=$which_epoch"
  local jid
  jid=$(_submit_parsable --job-name="$job_name" --exclude="$EXCLUDE" \
    "${dep_args[@]}" --export="$export_vars" "$TEST_SH")
  echo "test  $results_name epoch=$which_epoch -> job $jid ($dep_note)"
}

_local_epochs_for_run() {
  local run="$1"
  local save="${ORG_SAVE_EPOCHS[$run]//:/,}"
  local ep
  IFS=',' read -r -a _eps <<< "$save"
  for ep in "${_eps[@]}"; do
    [[ -n "$ep" ]] && echo "$ep"
  done
}

for organ in prostate kidney uterus; do
  while read -r ep; do
    _submit_fid c13_oa_all "$organ" "$ep"
  done < <(_local_epochs_for_run c13_oa_all)
  while read -r ep; do
    _submit_fid c13_oa_top3 "$organ" "$ep"
  done < <(_local_epochs_for_run c13_oa_top3)
done

while read -r ep; do
  _submit_fid c13_os_prostate prostate "$ep"
done < <(_local_epochs_for_run c13_os_prostate)

while read -r ep; do
  _submit_fid c13_os_kidney kidney "$ep"
done < <(_local_epochs_for_run c13_os_kidney)

while read -r ep; do
  _submit_fid c13_os_uterus uterus "$ep"
done < <(_local_epochs_for_run c13_os_uterus)

echo "Done. Train job ids (last hop): oa_all=${J_TRAIN[c13_oa_all]:-none} oa_top3=${J_TRAIN[c13_oa_top3]:-none} os_p=${J_TRAIN[c13_os_prostate]:-none} os_k=${J_TRAIN[c13_os_kidney]:-none} os_u=${J_TRAIN[c13_os_uterus]:-none}"
