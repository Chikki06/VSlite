#!/bin/bash
# Submit integer quantization sweep: int32 baseline + int4 + int8 (parallel).
# Login-node driver only (calls sbatch for quant/test). No GPU needed here.
# Each quant job auto-queues its test via _quantize_body.sh (afterok on quant).
#
# Usage (from repo root or any cwd):
#   ./jobs/cislnet/quant/quantize_int_sweep.sh
#   RUN=c13 ./jobs/cislnet/quant/quantize_int_sweep.sh

set -euo pipefail
: "${VSLITE_ROOT:=/projects/illinois/eng/bioe/rxb/akshat/VSlite}"
# shellcheck source=../../_lib/slurm_resolve.sh
source "$VSLITE_ROOT/jobs/_lib/slurm_resolve.sh"
VSLITE_ROOT="$(slurm_resolve_include_root jobs/_lib/vslite_env.sh "${BASH_SOURCE[0]:-$0}")" || exit 1

: "${RUN:=c13}"
: "${JOBS_DIR:=$VSLITE_ROOT/jobs/cislnet/quant}"

export RUN

J_INT32=$(sbatch --parsable "$JOBS_DIR/quantize_int32.sh")
J_INT4=$(sbatch --parsable "$JOBS_DIR/quantize_int4.sh")
J_INT8=$(sbatch --parsable "$JOBS_DIR/quantize_int8.sh")

echo "Submitted int quant sweep: int32=$J_INT32 int4=$J_INT4 int8=$J_INT8"
echo "Tests chained afterok from each quant job via _quantize_body.sh"
