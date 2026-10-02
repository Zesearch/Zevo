#!/usr/bin/env bash
# Zevo remote Python environments for a direct GPU host (cloud / instance).
#
# System-owned helper. The acquisition helper uploads it to ~/zevo/env/ and
# starts `build all` in the background right after the GPU health gate, so the
# environments are usually ready before the first GPU stage asks for them.
# Inference and Train then run `ensure <profile>`, which returns at once when
# the profile is ready, waits on a build in progress, or builds it itself.
#
# Before this, every agent activation wrote its own install sequence per host:
# run a676d8ac installed numpy three times with conflicting pins in Train and
# spent 10 of 16 candidate-inference minutes building a venv on a cold host.
#
#   remote_env.sh build  <infer|train|all>   build if missing (idempotent)
#   remote_env.sh ensure <infer|train>       block until ready; build if nobody is
#   remote_env.sh status                     JSON of every profile
#   remote_env.sh manifest <profile>         the pinned requirements
#   remote_env.sh python <profile>           print the interpreter path
#
# Layout: ~/zevo/env/<profile>/  (venv), .ready (manifest hash + versions),
#         .building (lock, pid), build.log.  Manifest hash in the .ready file
#         means a changed manifest rebuilds on the next ensure.
set -uo pipefail

ZEVO_ENV_ROOT="${ZEVO_ENV_ROOT:-$HOME/zevo/env}"
PY="${ZEVO_ENV_PYTHON:-python3}"

# Pinned stacks. Inference and Train share torch/transformers so one host can
# hold both without a conflict; TRL is only in the train profile.
manifest() {
  case "$1" in
    infer)
      cat <<'REQ'
vllm==0.10.2
transformers==4.56.1
datasets==4.0.0
pyyaml>=6
REQ
      ;;
    train)
      cat <<'REQ'
torch==2.8.0
transformers==4.56.1
trl==0.21.0
peft==0.17.1
accelerate==1.10.1
datasets==4.0.0
pyyaml>=6
REQ
      ;;
    *) echo "unknown profile: $1" >&2; return 2 ;;
  esac
}

manifest_hash() { manifest "$1" | { sha256sum 2>/dev/null || shasum -a 256; } | cut -c1-16; }
valid_profile() { case "$1" in infer|train) return 0 ;; *) echo "unknown profile: $1" >&2; return 2 ;; esac; }
profile_dir() { echo "$ZEVO_ENV_ROOT/$1"; }

is_ready() {
  local d; d="$(profile_dir "$1")"
  [ -f "$d/.ready" ] && [ "$(head -1 "$d/.ready" 2>/dev/null)" = "$(manifest_hash "$1")" ] \
    && [ -x "$d/bin/python" ]
}

build_one() {
  local p="$1" d; valid_profile "$p" || return 2
  d="$(profile_dir "$p")"
  mkdir -p "$d"
  if is_ready "$p"; then echo "READY $p (cached)"; return 0; fi
  # One builder per profile; a second caller waits in `ensure`.
  if [ -f "$d/.building" ] && kill -0 "$(cat "$d/.building" 2>/dev/null)" 2>/dev/null; then
    echo "BUILDING $p by pid $(cat "$d/.building")"; return 3
  fi
  echo $$ > "$d/.building"
  local t0; t0=$(date +%s)
  {
    echo "=== build $p $(date -u +%FT%TZ) manifest $(manifest_hash "$p")"
    rm -f "$d/.ready"
    if [ ! -x "$d/bin/python" ]; then
      "$PY" -m venv "$d" || { echo "VENV_FAILED"; exit 1; }
    fi
    "$d/bin/python" -m pip install --quiet --upgrade pip uv || echo "pip/uv upgrade failed; continuing with pip"
    manifest "$p" > "$d/requirements.txt"
    if ! "$d/bin/python" -m uv pip install --python "$d/bin/python" --quiet -r "$d/requirements.txt"; then
      echo "uv failed; falling back to pip"
      "$d/bin/python" -m pip install --quiet -r "$d/requirements.txt" || { echo "PIP_FAILED"; exit 1; }
    fi
    "$d/bin/python" - <<'PYCHK' | tee "$d/.versions" || { echo "IMPORT_FAILED"; exit 1; }
import importlib, json, sys
mods = {"torch": "torch", "transformers": "transformers"}
out = {}
for name in ("torch", "transformers", "vllm", "trl", "peft", "accelerate", "datasets"):
    try:
        m = importlib.import_module(name); out[name] = getattr(m, "__version__", "?")
    except Exception:
        pass
assert "torch" in out and "transformers" in out, out
import torch
out["cuda"] = torch.version.cuda or ""
out["cuda_available"] = bool(torch.cuda.is_available())
print("VERSIONS " + json.dumps(out))
PYCHK
  } > "$d/build.log" 2>&1
  local rc=$?
  rm -f "$d/.building"
  grep -q '^VERSIONS ' "$d/.versions" 2>/dev/null || rc=1
  if [ $rc -eq 0 ]; then
    { manifest_hash "$p"; grep '^VERSIONS ' "$d/build.log" | tail -1; } > "$d/.ready"
    echo "READY $p in $(( $(date +%s) - t0 ))s $(grep '^VERSIONS ' "$d/build.log" | tail -1)"
    return 0
  fi
  echo "FAILED $p rc=$rc; see $d/build.log"; tail -15 "$d/build.log"
  return 1
}

ensure() {
  local p="$1" d waited=0 limit="${ZEVO_ENV_WAIT_SECONDS:-1500}"
  valid_profile "$p" || return 2
  d="$(profile_dir "$p")"
  while :; do
    if is_ready "$p"; then echo "READY $p $d/bin/python"; return 0; fi
    if [ -f "$d/.building" ] && kill -0 "$(cat "$d/.building" 2>/dev/null)" 2>/dev/null; then
      [ $waited -ge "$limit" ] && { echo "TIMEOUT waiting for $p build (pid $(cat "$d/.building"))"; tail -5 "$d/build.log" 2>/dev/null; return 4; }
      sleep 10; waited=$((waited+10)); continue
    fi
    build_one "$p"; local rc=$?
    [ $rc -eq 3 ] && continue
    [ $rc -eq 0 ] && { echo "READY $p $d/bin/python"; return 0; }
    return $rc
  done
}

status() {
  local first=1; printf '{'
  for p in infer train; do
    local d; d="$(profile_dir "$p")"
    [ $first -eq 1 ] || printf ','; first=0
    if is_ready "$p"; then printf '"%s":{"ready":true,"python":"%s"}' "$p" "$d/bin/python"
    elif [ -f "$d/.building" ]; then printf '"%s":{"ready":false,"building":true}' "$p"
    else printf '"%s":{"ready":false}' "$p"; fi
  done
  printf '}\n'
}

case "${1:-}" in
  build)
    case "${2:-all}" in
      all) build_one infer; r1=$?; build_one train; r2=$?; [ $r1 -eq 0 ] && [ $r2 -eq 0 ] ;;
      *) build_one "$2" ;;
    esac ;;
  ensure) ensure "${2:?profile}" ;;
  status) status ;;
  manifest) valid_profile "${2:-}" && manifest "$2" ;;
  python) valid_profile "${2:-}" && echo "$(profile_dir "$2")/bin/python" ;;
  *) echo "usage: remote_env.sh build <infer|train|all> | ensure <profile> | status | manifest <profile> | python <profile>" >&2; exit 2 ;;
esac
