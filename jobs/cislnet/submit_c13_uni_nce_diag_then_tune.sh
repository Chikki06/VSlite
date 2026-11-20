#!/bin/bash
# Diagnose PatchNCE separation on c13_uni, then queue the targeted tune arms
# afterok on the diagnose job.
#
# Usage:
#   bash jobs/cislnet/submit_c13_uni_nce_diag_then_tune.sh
#   DRY_RUN=1 bash jobs/cislnet/submit_c13_uni_nce_diag_then_tune.sh
#   INCLUDE_LOW_T=0 EPOCHS=5 bash jobs/cislnet/submit_c13_uni_nce_diag_then_tune.sh

set -euo pipefail
VSLITE_ROOT="${VSLITE_ROOT:-/projects/illinois/eng/bioe/rxb/akshat/VSlite}"
cd "$VSLITE_ROOT"

DIAG_SH="$VSLITE_ROOT/jobs/cislnet/submit_c13_uni_nce_diag.sh"
TUNE_SH="$VSLITE_ROOT/jobs/cislnet/submit_c13_uni_nce_tune.sh"

echo "=== 1/2 diagnose (NCE margins + top-1) ==="
# Capture the last line (job id) from the diagnose submitter.
DIAG_OUT="$(bash "$DIAG_SH")"
echo "$DIAG_OUT"
DIAG_JID="$(echo "$DIAG_OUT" | tail -n1 | tr -d '[:space:]')"
if [[ -z "$DIAG_JID" ]]; then
  echo "failed to parse diagnose job id" >&2
  exit 2
fi

echo
echo "=== 2/2 tune arms (afterok:$DIAG_JID) ==="
DIAG_JID="$DIAG_JID" bash "$TUNE_SH"
