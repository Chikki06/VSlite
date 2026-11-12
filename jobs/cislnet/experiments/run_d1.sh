#!/bin/bash
# After A/B/C finish and tracking.csv is populated, queue the D combo grid.
# Prefer submit_d_combos.sh (top-K×top-K). This wrapper keeps the old entrypoint name.
set -euo pipefail
ROOT="/projects/illinois/eng/bioe/rxb/akshat/VSlite"
exec bash "$ROOT/jobs/cislnet/experiments/submit_d_combos.sh"
