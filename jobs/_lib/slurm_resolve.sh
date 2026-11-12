# shellcheck shell=bash
# Soft-fail resolver. Prefer the real VSlite checkout (has scripts/), not $HOME
# when $HOME/jobs is only a symlink into the project (would yield /u/akshahi2
# without scripts/train.py).
#
# Usage:
#   : "${VSLITE_ROOT:=/projects/illinois/eng/bioe/rxb/akshat/VSlite}"
#   source "$VSLITE_ROOT/jobs/_lib/slurm_resolve.sh"
#   VSLITE_ROOT="$(slurm_resolve_include_root jobs/_lib/vslite_env.sh "${BASH_SOURCE[0]:-$0}")" || exit 1

_slurm_is_vslite_root() {
  local root="$1"
  [[ -f "$root/scripts/train.py" && -f "$root/jobs/_lib/vslite_env.sh" ]]
}

_slurm_chase_jobs_symlink() {
  local root="$1"
  local real_jobs real_root
  [[ -e "$root/jobs" ]] || return 1
  real_jobs="$(cd "$root/jobs" && pwd -P)" || return 1
  real_root="$(dirname "$real_jobs")"
  if _slurm_is_vslite_root "$real_root"; then
    echo "$real_root"
    return 0
  fi
  return 1
}

slurm_resolve_include_root() {
  local needle="$1"
  local caller="${2:-}"
  local d root chased
  local -a bases=(
    "$(dirname "$caller")"
    "${SLURM_JOB_INCLUDES_HOME:-}"
    "${SLURM_SUBMIT_DIR:-}"
    "${VSLITE_ROOT:-}"
    "/projects/illinois/eng/bioe/rxb/akshat/VSlite"
    "$HOME"
    "/u/akshahi2"
  )
  for d in "${bases[@]}"; do
    [[ -n "$d" && -f "$d/$needle" ]] || continue
    root="$(cd "$d" && pwd -P)" || continue
    if _slurm_is_vslite_root "$root"; then
      echo "$root"
      return 0
    fi
    if chased="$(_slurm_chase_jobs_symlink "$root")"; then
      echo "$chased"
      return 0
    fi
  done
  echo "slurm_resolve_include_root: cannot find real VSlite root for ${needle}" >&2
  return 1
}
