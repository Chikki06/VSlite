#!/bin/bash
# Submit full quantization pipeline:
#   int32 baseline + fp16 + fp8 + int4 + int8 quant jobs (parallel when resources allow)
# Login-node driver only (calls sbatch for quant/test/visual). No GPU needed here.
# Each quant auto-queues its own test via _quantize_body.sh (afterok on quant job).
# visual_compare runs after all quants finish.
#
# Usage (from repo root or any cwd):
#   ./jobs/cislnet/quant/quantize_all.sh
#   RUN=c13 ./jobs/cislnet/quant/quantize_all.sh

set -euo pipefail
: "${VSLITE_ROOT:=/projects/illinois/eng/bioe/rxb/akshat/VSlite}"
# shellcheck source=../../_lib/slurm_resolve.sh
source "$VSLITE_ROOT/jobs/_lib/slurm_resolve.sh"
VSLITE_ROOT="$(slurm_resolve_include_root jobs/_lib/vslite_env.sh "${BASH_SOURCE[0]:-$0}")" || exit 1

: "${RUN:=c13}"
: "${JOBS_DIR:=$VSLITE_ROOT/jobs/cislnet/quant}"

export RUN

J_INT32=$(sbatch --parsable "$JOBS_DIR/quantize_int32.sh")
J_FP16=$(sbatch --parsable "$JOBS_DIR/quantize_fp16.sh")
J_FP8=$(sbatch --parsable "$JOBS_DIR/quantize_fp8.sh")
J_INT4=$(sbatch --parsable "$JOBS_DIR/quantize_int4.sh")
J_INT8=$(sbatch --parsable "$JOBS_DIR/quantize_int8.sh")

sbatch --dependency=afterok:"$J_INT32":"$J_FP16":"$J_FP8":"$J_INT4":"$J_INT8" "$JOBS_DIR/visual_compare.sh"

echo "Submitted quant jobs: int32=$J_INT32 fp16=$J_FP16 fp8=$J_FP8 int4=$J_INT4 int8=$J_INT8"
echo "Each quant queues its own test; visual_compare chained after all quants."
