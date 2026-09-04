#!/bin/bash
# Deprecated wrapper — use quantize_int32.sh / quantize_fp16.sh / quantize_fp8.sh / quantize_int4.sh / quantize_int8.sh
echo "Use: ./jobs/cislnet/quant/quantize_int32.sh | quantize_fp16.sh | ... (sbatch each)" >&2
echo "Or login-node sweeps: ./jobs/cislnet/quant/quantize_int_sweep.sh | quantize_all.sh" >&2
exit 1
