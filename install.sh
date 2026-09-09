#!/usr/bin/env bash
# Install the `zevo` CLI on your PATH.
#
# The CLI itself lives INSIDE the dockerized stack; this drops a tiny shim
# (ops/scripts/zevo) onto your PATH. Running `zevo` opens an interactive command
# shell whose commands execute in the `backend` container, where the DB, API,
# secrets, and Agent driver CLIs are already wired.
#
#   ./install.sh            # install
#   ./install.sh --uninstall
#
# Requires: docker + the stack running (`docker compose up -d`). Nothing is
# installed into Python — the command is just the shim.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHIM="$REPO/ops/scripts/zevo"

# Pick the first bin dir that's already on PATH (no PATH edits needed), else
# fall back to /usr/local/bin.
BIN=""
for d in "$HOME/.local/bin" /usr/local/bin /opt/homebrew/bin; do
  case ":$PATH:" in *":$d:"*) BIN="$d"; break;; esac
done
BIN="${BIN:-/usr/local/bin}"
LINK="$BIN/zevo"

_link() {  # ln -sf, retrying with sudo if the dir isn't writable
  mkdir -p "$BIN" 2>/dev/null || true
  if ln -sf "$SHIM" "$LINK" 2>/dev/null; then return 0; fi
  echo "  (need elevated perms for $BIN — using sudo)"
  sudo ln -sf "$SHIM" "$LINK"
}

if [ "${1:-}" = "--uninstall" ]; then
  rm -f "$LINK" 2>/dev/null || sudo rm -f "$LINK"
  echo "uninstalled: removed $LINK"
  echo "note: if you also added an 'alias zevo=...' to your shell rc, delete that line too."
  exit 0
fi

chmod +x "$SHIM"
_link
echo "installed: $LINK -> $SHIM"

case ":$PATH:" in
  *":$BIN:"*) echo "ready — run:  zevo" ;;
  *) echo "add '$BIN' to PATH, then reopen your shell:";
     echo "  echo 'export PATH=\"$BIN:\$PATH\"' >> ~/.zshrc && source ~/.zshrc" ;;
esac
echo "tip: the stack must be up (docker compose up -d) for commands to run."
