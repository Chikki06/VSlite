# Resolve CISLNet channel knobs (source from train/test/wsi).
# Sets: NUM_CHANNELS, NUM_IR, IR_INDICES (may be empty), and optionally RUN/NAME defaults for Vis+/IR-only.
#
# Vis+ N = VIS RGB + N IR bands → input_nc = 3+N.
# IR-only N = N IR bands, no VIS → input_nc = N.
# Wavelength order 1650,1081,1765,1140,1547 → IR indices 8,0,9,1,6.
# shellcheck shell=bash

# Optional VIS-only shorthand: c13-style model on VIS RGB alone (input_nc=3, no IR).
if [[ "${VIS_ONLY:-0}" == "1" ]]; then
  NUM_CHANNELS=3
  NUM_IR=0
  IR_INDICES=""
  RUN="${RUN:-visonly}"
  NAME="${NAME:-$RUN}"
# Optional IR-only shorthand (Vis+ band subsets, no VIS): IRONLY=1|3|5
elif [[ -n "${IRONLY:-}" ]]; then
  case "$IRONLY" in
    1)
      NUM_CHANNELS=1
      NUM_IR=1
      IR_INDICES="${IR_CHANNEL_INDICES:-8}"
      RUN="${RUN:-ir1}"
      ;;
    3)
      NUM_CHANNELS=3
      NUM_IR=3
      IR_INDICES="${IR_CHANNEL_INDICES:-8,0,9}"
      RUN="${RUN:-ir3}"
      ;;
    5)
      NUM_CHANNELS=5
      NUM_IR=5
      IR_INDICES="${IR_CHANNEL_INDICES:-8,0,9,1,6}"
      RUN="${RUN:-ir5}"
      ;;
    *)
      echo "Unsupported IRONLY=$IRONLY (expected 1, 3, or 5)" >&2
      exit 1
      ;;
  esac
  NAME="${NAME:-$RUN}"
# Optional Vis+ shorthand: VISPLUS=1|3|5
elif [[ -n "${VISPLUS:-}" ]]; then
  case "$VISPLUS" in
    1)
      NUM_CHANNELS=4
      NUM_IR=1
      IR_INDICES="${IR_CHANNEL_INDICES:-8}"
      RUN="${RUN:-vp1}"
      ;;
    3)
      NUM_CHANNELS=6
      NUM_IR=3
      IR_INDICES="${IR_CHANNEL_INDICES:-8,0,9}"
      RUN="${RUN:-vp3}"
      ;;
    5)
      NUM_CHANNELS=8
      NUM_IR=5
      IR_INDICES="${IR_CHANNEL_INDICES:-8,0,9,1,6}"
      RUN="${RUN:-vp5}"
      ;;
    *)
      echo "Unsupported VISPLUS=$VISPLUS (expected 1, 3, or 5)" >&2
      exit 1
      ;;
  esac
  NAME="${NAME:-$RUN}"
else
  NUM_CHANNELS="${NUM_CHANNELS:-13}"
  IR_INDICES="${IR_CHANNEL_INDICES:-}"
  case "$NUM_CHANNELS" in
    1)  IR_INDICES="${IR_INDICES:-8}"; NUM_IR=1 ;;             # IR-only Vis+1 bands
    3)  IR_INDICES="${IR_INDICES:-8,0,9}"; NUM_IR=3 ;;         # IR-only Vis+3 bands
    4)  IR_INDICES="${IR_INDICES:-8}"; NUM_IR=1 ;;             # Vis+1
    5)  IR_INDICES="${IR_INDICES:-8,0,9,1,6}"; NUM_IR=5 ;;     # IR-only Vis+5 bands
    6)
      # IR-only first-6 unless indices (or VISPLUS path above) select a Vis+ subset.
      if [[ -n "$IR_INDICES" ]]; then
        NUM_IR="$(awk -F',' '{print NF}' <<<"$IR_INDICES")"
      else
        NUM_IR=6
      fi
      ;;
    8)  IR_INDICES="${IR_INDICES:-8,0,9,1,6}"; NUM_IR=5 ;;   # Vis+5
    10) NUM_IR=10 ;;
    13) NUM_IR=10 ;;
    *)
      echo "Unsupported NUM_CHANNELS=$NUM_CHANNELS" >&2
      exit 1
      ;;
  esac
fi
