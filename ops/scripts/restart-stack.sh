#!/bin/bash
# One-shot: kill EVERY running ZEVO process / container, then bring up a
# single clean Docker Compose stack.
#
# Invariant after this script: exactly one of each running --
#   - one postgres (the container; the host's homebrew postgres on :5432
#     is left alone because it serves the user's other apps)
#   - one backend (container :8000, mapped to host port $BACKEND_PORT, default 8001)
#   - one scheduler (container)
#   - one web (container, host port $WEB_PORT, default 5173)
#
# Run from anywhere; it cd's to the repo.

set -euo pipefail
cd "$(dirname "$0")/../.."
REPO="$(pwd)"
echo "[restart] repo=$REPO"

# Host ports the stack publishes. Each must match docker-compose.yml's
# mapping defaults and honor an override in .env, so the port-free +
# health-check steps and the final URL all target the right ports.
env_port() {
  # Last assignment wins; strip quotes/spaces and inline comments.
  grep -E "^$1=" .env 2>/dev/null | tail -1 | cut -d= -f2 | cut -d'#' -f1 | tr -d " '\"" || true
}
BACKEND_PORT="$(env_port BACKEND_PORT)";  BACKEND_PORT="${BACKEND_PORT:-8001}"
WEB_PORT="$(env_port WEB_PORT)";          WEB_PORT="${WEB_PORT:-5173}"
POSTGRES_PORT="$(env_port POSTGRES_PORT)"; POSTGRES_PORT="${POSTGRES_PORT:-55432}"
echo "[restart] ports: backend=$BACKEND_PORT web=$WEB_PORT postgres=$POSTGRES_PORT"

# ---------- 1. tear down this compose project only ----------
if docker info >/dev/null 2>&1; then
  echo "[restart] docker compose down --remove-orphans"
  docker compose down --remove-orphans 2>&1 | tail -5 || true
else
  echo "[restart] Docker daemon not running -- start Docker Desktop first" >&2
  exit 1
fi

# ---------- 2. refuse to kill unrelated processes that own our ports ----------
for port in "$WEB_PORT" "$BACKEND_PORT" "$POSTGRES_PORT"; do
  pids=$(lsof -nP -iTCP:$port -sTCP:LISTEN -t 2>/dev/null || true)
  if [ -n "$pids" ]; then
    echo "[restart] port $port is still owned by PID(s) $pids; stop that process or choose another port" >&2
    exit 1
  fi
done

# ---------- 3. make sure .env exists ----------
if [ ! -f .env ]; then
  cat > .env <<EOF
POSTGRES_USER=zevo
POSTGRES_PASSWORD=zevo
POSTGRES_DB=zevo_dev
POSTGRES_PORT=55432
EOF
  chmod 600 .env
  echo "[restart] wrote default .env"
fi

# ---------- 4. up ----------
echo "[restart] docker compose up -d --build"
docker compose up -d --build

# ---------- 5. wait for backend health ----------
echo -n "[restart] waiting for backend health "
healthy=0
for i in {1..60}; do
  if curl -sf "http://localhost:${BACKEND_PORT}/health" >/dev/null 2>&1; then
    echo " OK after ${i}s"
    healthy=1
    break
  fi
  echo -n "."
  sleep 1
done
if [ "$healthy" -ne 1 ]; then
  echo " FAILED after 60s"
  echo "[restart] backend never became healthy; recent logs:" >&2
  docker compose logs --tail 40 backend >&2 || true
  exit 1
fi

# ---------- 6. report ----------
echo
docker compose ps
echo
echo "[restart] open http://localhost:${WEB_PORT}"
