# shellcheck shell=bash
# Slurm runs a copy of the batch script under /var/spool/slurmd/.../slurm_script.
# Do not use dirname "${BASH_SOURCE[0]}" alone to find repo files (jobs/_lib/...).
#
# Usage:
#   : "${VSLITE_ROOT:=/projects/illinois/eng/bioe/rxb/akshat/VSlite}"
#   source "$VSLITE_ROOT/jobs/_lib/slurm_resolve.sh"
#   VSLITE_ROOT="$(slurm_resolve_include_root jobs/_lib/vslite_env.sh "${BASH_SOURCE[0]:-$0}")" || exit 1
#   source "$VSLITE_ROOT/jobs/_lib/vslite_env.sh"

slurm_resolve_include_root() {
  local needle="$1"
  local caller="${2:-}"
  local d bases

  bases=(
    "$(dirname "$caller")"
    "${SLURM_JOB_INCLUDES_HOME:-}"
    "${SLURM_SUBMIT_DIR:-}"
    "${VSLITE_ROOT:-}"
    "$HOME"
    "/u/akshahi2"
    "/projects/illinois/eng/bioe/rxb/akshat/VSlite"
  )
  for d in "${bases[@]}"; do
    [[ -n "$d" && -f "$d/$needle" ]] || continue
    (cd "$d" && pwd)
    return 0
  done
  echo "slurm_resolve_include_root: cannot find ${needle} (submit from repo root or set SLURM_JOB_INCLUDES_HOME)" >&2
  return 1
}
