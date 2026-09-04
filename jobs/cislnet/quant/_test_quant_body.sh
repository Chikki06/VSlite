# Test-quant entrypoint (source after vslite_env + _quant_common).
# Must be sourced via absolute path: $VSLITE_ROOT/jobs/cislnet/quant/_test_quant_body.sh

QUANT_CKPT="$QUANT_ROOT/$QUANT_PRECISION/net_G_${QUANT_PRECISION}.pt"
EXTRA=()
[[ "${SAVE_TRIPLETS:-0}" == "1" ]] && EXTRA+=(--save_triplets)

python "$VSLITE_ROOT/scripts/vs_infer_and_metrics_quant.py" \
  --quant_ckpt "$QUANT_CKPT" \
  --opt_dir "$OPT_DIR" \
  --source_ckpt "$SOURCE_CKPT" \
  --precision "$QUANT_PRECISION" \
  --dataroot "$DATAROOT" \
  --results_dir "$RESULTS_DIR" \
  --run "$RUN" \
  --gpu_ids 0 \
  --split test \
  --num_test "$NUM_TEST" \
  --input_nc 13 \
  --num_ir_channels 10 \
  --fid_max_items "${VSLITE_FID_MAX_ITEMS:-$NUM_TEST}" \
  --fid_batch_size "${VSLITE_FID_BATCH_SIZE:-64}" \
  "${EXTRA[@]}"
