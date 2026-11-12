#!/bin/bash
# Queue Exp0 then Groups A/B/C (D1 after select_pareto_d1.py).
# Train jobs are two hops: hop2 afterany + CONTINUE_TRAIN=true so a 40h
# walltime timeout still finishes EPOCHS (default 20 via _common.sh).
set -euo pipefail
ROOT="/projects/illinois/eng/bioe/rxb/akshat/VSlite"
EPOCHS="${EPOCHS:-20}"
TRAIN_CHAIN="${TRAIN_CHAIN:-1}"
EXCLUDE="${EXCLUDE:-ccc0388}"

mkdir -p /u/akshahi2/logs/cn_exp0 /u/akshahi2/logs/cn_a1 /u/akshahi2/logs/cn_a2 \
  /u/akshahi2/logs/cn_a3 /u/akshahi2/logs/cn_a4 /u/akshahi2/logs/cn_b1 \
  /u/akshahi2/logs/cn_b2 /u/akshahi2/logs/cn_b3 /u/akshahi2/logs/cn_c1 /u/akshahi2/logs/cn_d1

_submit() {
  sbatch --parsable --exclude="$EXCLUDE" --export=ALL,EPOCHS="$EPOCHS" "$@"
}

_submit_train_chained() {
  local script="$1"
  local jid
  jid=$(_submit "$script")
  echo "$(basename "$script") hop1 -> $jid"
  if [[ "$TRAIN_CHAIN" == "1" ]]; then
    local jid2
    jid2=$(_submit --dependency="afterany:${jid}" --export=ALL,EPOCHS="$EPOCHS",CONTINUE_TRAIN=true "$script")
    echo "$(basename "$script") hop2 -> $jid2 (afterany:$jid CONTINUE_TRAIN=true)"
  fi
}

sbatch --exclude="$EXCLUDE" "$ROOT/jobs/cislnet/experiments/exp0_baseline_metrics.sh"
_submit_train_chained "$ROOT/jobs/cislnet/experiments/a1_coordatt.sh"
_submit_train_chained "$ROOT/jobs/cislnet/experiments/a2_triplet.sh"
_submit_train_chained "$ROOT/jobs/cislnet/experiments/a3_scse.sh"
_submit_train_chained "$ROOT/jobs/cislnet/experiments/a4_gcblock.sh"
_submit_train_chained "$ROOT/jobs/cislnet/experiments/b1_half.sh"
_submit_train_chained "$ROOT/jobs/cislnet/experiments/b2_scconv.sh"
_submit_train_chained "$ROOT/jobs/cislnet/experiments/b3_convnext.sh"
_submit_train_chained "$ROOT/jobs/cislnet/experiments/c1_linear.sh"
echo "Submitted Exp0 + A1-A4 + B1-B3 + C1 (EPOCHS=$EPOCHS TRAIN_CHAIN=$TRAIN_CHAIN). After they finish:"
echo "  python $ROOT/scripts/select_pareto_d1.py && sbatch $ROOT/jobs/cislnet/experiments/d1_combine.sh"
