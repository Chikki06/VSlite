# Quantize entrypoint (source after vslite_env + _quant_common).
# Must be sourced via absolute path: $VSLITE_ROOT/jobs/cislnet/quant/_quantize_body.sh

_queue_quant_test() {
  if [[ "${SUBMIT_TEST:-1}" != "1" || -z "${SLURM_JOB_ID:-}" ]]; then
    return 0
  fi
  local test_script="$VSLITE_ROOT/jobs/cislnet/quant/test_${QUANT_PRECISION}.sh"
  if [[ ! -f "$test_script" ]]; then
    echo "No test script for QUANT_PRECISION=${QUANT_PRECISION} (skip auto-submit)" >&2
    return 0
  fi
  local test_jid
  test_jid=$(sbatch --parsable --dependency=afterok:"${SLURM_JOB_ID}" "$test_script")
  echo "Queued test ${QUANT_PRECISION} job ${test_jid} after quant ${SLURM_JOB_ID}"
}

_queue_quant_test

OUT_DIR="$QUANT_ROOT/$QUANT_PRECISION"
EXTRA=()
if [[ "$QUANT_PRECISION" == "fp8" && "${FP8_STATIC:-0}" == "1" ]]; then
  EXTRA+=(--fp8_static)
fi
if [[ "${QUANT_USE_ACT_SCALE:-0}" == "1" ]]; then
  EXTRA+=(--use_act_scale)
fi
if [[ "${QUANT_SKIP_CBAM:-0}" == "1" ]]; then
  EXTRA+=(--skip_cbam)
fi
if [[ -n "${QUANT_CALIB_PERCENTILE:-}" && "${QUANT_CALIB_PERCENTILE}" != "100" ]]; then
  EXTRA+=(--calib_percentile "$QUANT_CALIB_PERCENTILE")
fi
if [[ "${QUANT_FPGA_EXPORT:-0}" == "1" ]]; then
  EXTRA+=(--fpga_export)
fi
if [[ "$QUANT_PRECISION" == "fp8" && "${QUANT_FP8_NATIVE:-1}" == "0" ]]; then
  EXTRA+=(--fp8_native 0)
fi

python "$VSLITE_ROOT/scripts/quantize_cislnet.py" \
  --source_ckpt "$SOURCE_CKPT" \
  --opt_dir "$OPT_DIR" \
  --precision "$QUANT_PRECISION" \
  --out_dir "$OUT_DIR" \
  --dataroot "$DATAROOT" \
  --calib_num_patches "$CALIB_NUM_PATCHES" \
  --calib_batch_size "$CALIB_BATCH_SIZE" \
  --gpu_ids 0 \
  "${EXTRA[@]}"

echo "Quantized $QUANT_PRECISION -> $OUT_DIR"
