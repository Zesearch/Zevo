#!/usr/bin/env bash
# Copy the canonical AI-loop workflows and prompts into a repo clone.
# This directory is the source of truth; the two repos are rendered from it.
#
#   ./sync.sh ../../Zevo-ZeroToEvolved

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:?usage: sync.sh <path to repo clone>}"
# -e not -d: in a linked worktree .git is a file pointing at the real gitdir.
[ -e "$DEST/.git" ] || { echo "not a git checkout: $DEST" >&2; exit 1; }

mkdir -p "$DEST/.github/workflows" "$DEST/.github/ai-loop/prompts"
cp "$HERE"/workflows/ai-*.yml "$DEST/.github/workflows/"
cp "$HERE"/prompts/*.md       "$DEST/.github/ai-loop/prompts/"
cp "$HERE"/bootstrap.sh "$HERE"/sync.sh "$HERE"/README.md "$DEST/.github/ai-loop/"

echo "Synced into $DEST"
git -C "$DEST" status --short .github
