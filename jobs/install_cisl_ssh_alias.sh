#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="$SCRIPT_DIR/ssh_config_cisl_inference.template"
SSH_DIR="${HOME}/.ssh"
SSH_CONFIG="$SSH_DIR/config"

mkdir -p "$SSH_DIR"
chmod 700 "$SSH_DIR"
touch "$SSH_CONFIG"
chmod 600 "$SSH_CONFIG"

if grep -qE '^Host[[:space:]]+cisl-inference$' "$SSH_CONFIG"; then
  echo "Host alias cisl-inference already exists in $SSH_CONFIG"
  exit 0
fi

{
  echo
  echo "# Added by install_cisl_ssh_alias.sh on $(date -Is)"
  cat "$TEMPLATE"
} >> "$SSH_CONFIG"

echo "Added cisl-inference host alias to $SSH_CONFIG"
echo "Update IdentityFile in $SSH_CONFIG if your key path differs."
