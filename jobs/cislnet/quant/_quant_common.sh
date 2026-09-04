# Shared CISLNet quantization job settings (source from quant/*.sh).
# shellcheck disable=SC2034

: "${VSLITE_ROOT:=/projects/illinois/eng/bioe/rxb/akshat/VSlite}"
: "${RUN:=c13}"
: "${RESULTS_DIR:=$VSLITE_ROOT/results/cislnet}"
: "${SOURCE_CKPT:=$RESULTS_DIR/$RUN/latest_net_G.pth}"
: "${OPT_DIR:=$RESULTS_DIR/$RUN}"
: "${QUANT_ROOT:=$RESULTS_DIR/quant/$RUN}"
: "${QUANT_PRECISION:=fp16}"
: "${CALIB_BATCH_SIZE:=8}"
: "${PATCH_SIZE:=1024}"
: "${NUM_TEST:=75000}"
: "${NUM_PATCHES:=24}"
: "${SUBMIT_TEST:=1}"

# Match CISLNet train resolution for calibration forwards.
export PATCH_SIZE

# Integer PTQ uses more train patches; fp8 e4m3fn uses a smaller calibration set.
case "${QUANT_PRECISION}" in
  fp8)
    : "${CALIB_NUM_PATCHES:=512}"
    export FP8_STATIC="${FP8_STATIC:-0}"
    export QUANT_FP8_NATIVE="${QUANT_FP8_NATIVE:-1}"
    pip install -q 'torchao>=0.17' 2>/dev/null || true
    export VSLITE_INFER_BS="${VSLITE_INFER_BS:-32}"
    export VSLITE_FID_BATCH_SIZE="${VSLITE_FID_BATCH_SIZE:-128}"
    ;;
  int4|int8)
    : "${CALIB_NUM_PATCHES:=1024}"
    export QUANT_USE_ACT_SCALE="${QUANT_USE_ACT_SCALE:-0}"
    export QUANT_STORE_PARAMS_FP32="${QUANT_STORE_PARAMS_FP32:-1}"
    export VSLITE_INFER_BS="${VSLITE_INFER_BS:-32}"
    export VSLITE_FID_BATCH_SIZE="${VSLITE_FID_BATCH_SIZE:-128}"
    ;;
  fp16)
    : "${CALIB_NUM_PATCHES:=512}"
    export VSLITE_INFER_BS="${VSLITE_INFER_BS:-16}"
    export VSLITE_FID_BATCH_SIZE="${VSLITE_FID_BATCH_SIZE:-64}"
    ;;
  int32)
    : "${CALIB_NUM_PATCHES:=0}"
    export VSLITE_INFER_BS="${VSLITE_INFER_BS:-16}"
    export VSLITE_FID_BATCH_SIZE="${VSLITE_FID_BATCH_SIZE:-64}"
    ;;
  *)
    : "${CALIB_NUM_PATCHES:=512}"
    ;;
esac

export PYTHONUNBUFFERED=1
