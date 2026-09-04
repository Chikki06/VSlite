# Slurm-safe bootstrap. Always source this file via its absolute repo path.
set -euo pipefail
_SCRIPT="${BASH_SOURCE[0]}"
if [[ "$_SCRIPT" == /* ]]; then
  VSLITE_ROOT="$(cd "$(dirname "$_SCRIPT")/../../.." && pwd)"
else
  VSLITE_ROOT="${VSLITE_ROOT:-/projects/illinois/eng/bioe/rxb/akshat/VSlite}"
fi
# shellcheck source=../../_lib/vslite_env.sh
source "$VSLITE_ROOT/jobs/_lib/vslite_env.sh"
# shellcheck source=_quant_common.sh
source "$VSLITE_ROOT/jobs/cislnet/quant/_quant_common.sh"
