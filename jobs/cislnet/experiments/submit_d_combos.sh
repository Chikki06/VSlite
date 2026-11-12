#!/bin/bash
# Select top-K×top-K D combos and queue each with a 2-hop train chain.
# Usage:
#   bash jobs/cislnet/experiments/submit_d_combos.sh
#   TOP_K=2 EPOCHS=20 bash jobs/cislnet/experiments/submit_d_combos.sh
set -euo pipefail
ROOT="/projects/illinois/eng/bioe/rxb/akshat/VSlite"
EPOCHS="${EPOCHS:-20}"
TOP_K="${TOP_K:-2}"
TRAIN_CHAIN="${TRAIN_CHAIN:-1}"
EXCLUDE="${EXCLUDE:-ccc0388}"
COMBOS_DIR="${COMBOS_DIR:-$ROOT/results/cislnet/experiments/d_combos}"

python "$ROOT/scripts/select_pareto_d1.py" --top_k "$TOP_K" --rewrite_tracking_groups
mkdir -p /u/akshahi2/logs/cn_d1

_submit() {
  sbatch --parsable --exclude="$EXCLUDE" "$@"
}

mapfile -t ENVS < <(ls -1 "$COMBOS_DIR"/*.sh 2>/dev/null | sort)
if [[ ${#ENVS[@]} -eq 0 ]]; then
  echo "FATAL: no combo env files in $COMBOS_DIR" >&2
  exit 1
fi

echo "Queuing ${#ENVS[@]} D-combo runs (EPOCHS=$EPOCHS TRAIN_CHAIN=$TRAIN_CHAIN)"
for envf in "${ENVS[@]}"; do
  # shellcheck disable=SC1090
  source "$envf"
  job_name="cn_d1_${NAME}"
  export_vars="ALL,EPOCHS=${EPOCHS},PARETO_ENV=${envf}"
  jid=$(_submit \
    --job-name="$job_name" \
    --export="$export_vars" \
    "$ROOT/jobs/cislnet/experiments/d1_combine.sh")
  echo "  $NAME hop1 -> $jid  (env=$(basename "$envf"))"
  if [[ "$TRAIN_CHAIN" == "1" ]]; then
    jid2=$(_submit \
      --job-name="$job_name" \
      --dependency="afterany:${jid}" \
      --export="${export_vars},CONTINUE_TRAIN=true" \
      "$ROOT/jobs/cislnet/experiments/d1_combine.sh")
    echo "  $NAME hop2 -> $jid2 (afterany:$jid)"
  fi
done
