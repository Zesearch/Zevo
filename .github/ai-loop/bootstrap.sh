#!/usr/bin/env bash
# Create the AI-loop labels and secrets on a repo. Safe to re-run.
#
#   ./bootstrap.sh <owner>/<repo>
#
# Secrets are read from the environment so they never land in shell history:
#   CLAUDE_OAUTH_IMPLEMENTER  account A, writes code and answers review comments
#   CLAUDE_OAUTH_REVIEWER     account B, reviews
#   CLAUDE_OAUTH_FAILOVER     account C, retries whichever of the two rate-limits
#
# Generate each one on the matching Claude Max login with: claude setup-token

set -euo pipefail
REPO="${1:?usage: bootstrap.sh <owner/repo>}"

label() {
  gh label create "$1" -R "$REPO" --color "$2" --description "$3" --force >/dev/null
  echo "  label $1"
}

echo "Labels on $REPO"
label "ai:skip"               "cfd3d7" "Do not let the AI loop pick this issue up"
label "ai:implementing"       "1d76db" "Implementer is turning this issue into a PR"
label "ai:reviewing"          "5319e7" "Reviewer is reading this PR"
label "ai:addressing"         "0e8a16" "Implementer is answering review comments"
label "ai:changes-requested"  "d93f0b" "Reviewer wants changes; another round is queued"
label "ai:approved"           "0e8a16" "Reviewer is satisfied; ready for a human to merge"
label "ai:needs-human"        "b60205" "Loop stopped and needs a person"
for n in 1 2 3 4; do label "ai:round-$n" "ededed" "Review round $n"; done

echo "Secrets on $REPO"
for name in CLAUDE_OAUTH_IMPLEMENTER CLAUDE_OAUTH_REVIEWER CLAUDE_OAUTH_FAILOVER; do
  value="${!name:-}"
  if [ -z "$value" ]; then
    echo "  SKIP $name (not set in the environment)"
    continue
  fi
  printf '%s' "$value" | gh secret set "$name" -R "$REPO" >/dev/null
  echo "  set  $name"
done

echo "Done. The three ai-*.yml workflows must be on the default branch before the loop can chain."
