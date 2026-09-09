#!/bin/bash
# Cloud-init user-data for the ZEVO EC2 host.
#
# Brings up Docker + the full zevo stack (postgres + backend + scheduler +
# web) on first boot. Target AMI: Ubuntu 24.04 LTS.
#
# Cloud-init user-data has no separate variables channel: set the values by
# editing the export lines into a COPY of this script (or prepending
# `export VAR=...` lines) before passing it as user-data:
#   ZEVO_REPO_URL  -- https://github.com/Zesearch/Zevo-ZeroToEvolved.git
#   ZEVO_BRANCH    -- main (or another branch/ref to deploy)
#   CLAUDE_CODE_OAUTH_TOKEN          -- Claude CLI (Max/Pro), from `claude setup-token`
#   AWS_BEARER_TOKEN_BEDROCK         -- Bedrock API key (bearer token), for the bedrock driver
#   AWS_REGION                       -- e.g. us-east-1
#   OPENAI_API_KEY / OPENROUTER_API_KEY (optional) -- usage-billed alternate agent drivers
# Personal Codex subscriptions are initialized interactively after boot with
# `docker compose exec scheduler codex login --device-auth`; never paste a
# browser-session token into this script or .env.
#   ANTHROPIC_AUTH_TOKEN / ANTHROPIC_API_KEY (optional) -- claude_cli fallback auth
#   VASTAI_API_KEY (optional)        -- Vast.ai cloud GPUs
#   LAMBDA_API_KEY / LAMBDA_SSH_KEY_NAME (optional) -- Lambda Cloud GPUs
#   ZEVO_CLUSTER_* / ZEVO_INSTANCE_* (optional)     -- your own Slurm machines

set -euo pipefail

LOG=/var/log/zevo-bootstrap.log
exec > >(tee -a "$LOG") 2>&1
echo "[$(date -Iseconds)] zevo-bootstrap start"

ZEVO_USER="ubuntu"
ZEVO_HOME="/home/$ZEVO_USER"
ZEVO_REPO_URL="${ZEVO_REPO_URL:-https://github.com/Zesearch/Zevo-ZeroToEvolved.git}"
ZEVO_BRANCH="${ZEVO_BRANCH:-main}"
ZEVO_DIR="$ZEVO_HOME/Zevo-ZeroToEvolved"

# --- 1. OS deps -----------------------------------------------------------
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y \
    ca-certificates curl gnupg lsb-release git tmux jq \
    openssh-client

# Docker (official)
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
  > /etc/apt/sources.list.d/docker.list
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

usermod -aG docker "$ZEVO_USER"

# --- 2. Swap (only matters on small instances; t3.small with 2 GB benefits) ---
if ! swapon --show | grep -q '/swapfile'; then
  fallocate -l 4G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

# --- 3. Clone repo --------------------------------------------------------
sudo -u "$ZEVO_USER" bash -c "
  set -euo pipefail
  cd '$ZEVO_HOME'
  if [ ! -d '$ZEVO_DIR/.git' ]; then
    git clone --branch '$ZEVO_BRANCH' '$ZEVO_REPO_URL' '$ZEVO_DIR'
  else
    cd '$ZEVO_DIR'
    git fetch origin
    git checkout '$ZEVO_BRANCH'
    git pull --ff-only
  fi
"

# --- 4. .env for compose --------------------------------------------------
# ZEVO_HOST_REPO / ZEVO_HOST_HOME are REQUIRED here: the systemd unit runs
# compose without ${PWD}/${HOME}, and without these the SSH/codex mounts
# would resolve to the host root. POSTGRES_PORT is pinned so the DB is only
# reachable where the compose default promises (55432).
ENV_FILE="$ZEVO_DIR/.env"
cat > "$ENV_FILE" <<EOF
POSTGRES_USER=zevo
POSTGRES_PASSWORD=zevo
POSTGRES_DB=zevo_dev
POSTGRES_PORT=55432
ZEVO_HOST_REPO=$ZEVO_DIR
ZEVO_HOST_HOME=$ZEVO_HOME
CLAUDE_CODE_OAUTH_TOKEN=${CLAUDE_CODE_OAUTH_TOKEN:-}
ANTHROPIC_AUTH_TOKEN=${ANTHROPIC_AUTH_TOKEN:-}
ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}
OPENAI_API_KEY=${OPENAI_API_KEY:-}
OPENROUTER_API_KEY=${OPENROUTER_API_KEY:-}
AWS_BEARER_TOKEN_BEDROCK=${AWS_BEARER_TOKEN_BEDROCK:-}
AWS_REGION=${AWS_REGION:-us-east-1}
VASTAI_API_KEY=${VASTAI_API_KEY:-}
LAMBDA_API_KEY=${LAMBDA_API_KEY:-}
LAMBDA_SSH_KEY_NAME=${LAMBDA_SSH_KEY_NAME:-}
ZEVO_CLOUD_BACKEND=${ZEVO_CLOUD_BACKEND:-}
LAMBDA_SSH_KEY_PATH=${LAMBDA_SSH_KEY_PATH:-}
# gpu_provider=cluster / instance — fill in to use your own Slurm machines.
ZEVO_CLUSTER_SSH_HOST=${ZEVO_CLUSTER_SSH_HOST:-}
ZEVO_CLUSTER_SSH_PORT=${ZEVO_CLUSTER_SSH_PORT:-22}
ZEVO_CLUSTER_SSH_USER=${ZEVO_CLUSTER_SSH_USER:-}
ZEVO_CLUSTER_SSH_KEY=${ZEVO_CLUSTER_SSH_KEY:-}
ZEVO_CLUSTER_ENV_SETUP=${ZEVO_CLUSTER_ENV_SETUP:-}
ZEVO_CLUSTER_REMOTE_DIR=${ZEVO_CLUSTER_REMOTE_DIR:-}
ZEVO_CLUSTER_SLURM_PARTITION=${ZEVO_CLUSTER_SLURM_PARTITION:-}
ZEVO_CLUSTER_SLURM_ACCOUNT=${ZEVO_CLUSTER_SLURM_ACCOUNT:-}
ZEVO_CLUSTER_SLURM_QOS=${ZEVO_CLUSTER_SLURM_QOS:-}
ZEVO_INSTANCE_SSH_HOST=${ZEVO_INSTANCE_SSH_HOST:-}
ZEVO_INSTANCE_SSH_PORT=${ZEVO_INSTANCE_SSH_PORT:-22}
ZEVO_INSTANCE_SSH_USER=${ZEVO_INSTANCE_SSH_USER:-}
ZEVO_INSTANCE_SSH_KEY=${ZEVO_INSTANCE_SSH_KEY:-}
ZEVO_INSTANCE_ENV_SETUP=${ZEVO_INSTANCE_ENV_SETUP:-}
ZEVO_INSTANCE_REMOTE_DIR=${ZEVO_INSTANCE_REMOTE_DIR:-}
EOF
chown "$ZEVO_USER:$ZEVO_USER" "$ENV_FILE"
chmod 600 "$ENV_FILE"

# --- 5. Install systemd unit + start --------------------------------------
cp "$ZEVO_DIR/ops/deploy/zevo.service" /etc/systemd/system/zevo.service
systemctl daemon-reload
systemctl enable zevo.service
systemctl start zevo.service

echo "[$(date -Iseconds)] zevo-bootstrap complete. Tail journal: journalctl -u zevo.service -f"
