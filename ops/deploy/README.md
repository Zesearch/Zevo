# Deploy

Two ways to bring up the full stack.

## 1. Local (Docker Compose)

```bash
# 0. Create the local config and fill only the providers you use.
cp .env.example .env
# Edit .env, or set credentials later in the UI under Settings → Secrets.

# 1. Start the stack.
docker compose up -d --build

# Personal ChatGPT subscription: `codex login` on this host is copied into the
# shared container credential volume on every start. OPENAI_API_KEY is used only
# when that login does not exist.

# 2. Open Zevo.
open http://localhost:5173       # dashboard
open http://localhost:8001/api/docs  # API
```

Services:
- `postgres` (host 55432 -> container 5432) -- backing DB; a host-level
  postgres on :5432 is left alone.
- `backend` (host 8001 -> container 8000) -- FastAPI + auto-applies Alembic migrations on boot.
- `scheduler` -- heartbeat daemon, picks up ready tickets.
- `web` (5173) -- React UI served by nginx; proxies `/api` and `/ws` to backend.

Runtime `data/`, `playbook/`, `src/zevo/`, `alembic/`, and `ops/` are
bind-mounted where needed. Agent-document and Python changes therefore reach
the backend/scheduler after a restart; rebuild `web` for UI changes.

## 2. EC2 (one host)

Target AMI: Ubuntu 24.04 LTS, t3.small or larger (2 GB RAM is enough since
GPU work happens on Vast.ai).

```bash
# launch with ops/deploy/ec2-bootstrap.sh as user-data
# (the ami id below is Ubuntu 24.04 LTS in us-east-1; look up your region's
#  current one: aws ssm get-parameter --name \
#  /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id)
aws ec2 run-instances \
  --image-id ami-0e2c8caa4b6378d8c \
  --instance-type t3.small \
  --key-name your-key \
  --security-group-ids sg-... \
  --user-data file://ops/deploy/ec2-bootstrap.sh
```

The user-data script:
1. Installs Docker.
2. Clones the repo at `ZEVO_BRANCH` (default `main`).
3. Writes `.env` from the supported driver/cloud environment variables.
4. Installs `ops/deploy/zevo.service` -- `docker compose up -d --build` on boot.

Reach the UI via SSH tunnel:
```bash
ssh -L 5173:localhost:5173 ubuntu@<ec2-ip>
# then open http://localhost:5173 in your local browser
```

The published ports listen on `127.0.0.1` only. For a personal ChatGPT
subscription, initialize Codex once on the EC2 host after the stack is up:

```bash
ssh ubuntu@<ec2-ip>
cd ~/Zevo-ZeroToEvolved
docker compose exec scheduler codex login --device-auth
docker compose restart backend scheduler holdout-scheduler
```

The login is stored in the shared `zevo_codex_home` volume. Do not put a token
copied from `auth.json` into `.env`; `OPENAI_API_KEY` is a separate, usage-billed
fallback.

Tail the stack:
```bash
journalctl -u zevo.service -f
docker compose logs -f backend scheduler holdout-scheduler
```

Stop / restart:
```bash
sudo systemctl stop zevo.service
sudo systemctl start zevo.service
```
