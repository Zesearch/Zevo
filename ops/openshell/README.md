# Running the ZEVO orchestrator inside NVIDIA OpenShell

OpenShell (https://build.nvidia.com/openshell) is an open-source secure runtime
for autonomous agents: per-agent sandboxes with kernel-level isolation (Landlock),
a default-deny network opened by a declarative YAML policy, credentials injected
as env vars (never onto the sandbox FS), and a full allow/deny audit trail. It
runs Claude Code **unmodified** — we just prefix the driver's command with
`openshell sandbox create … --`.

This wraps only the `claude_cli` **orchestrator** via
`src/zevo/engine/agent/drivers/sandbox.py`. Select OpenShell on that Agent's
Configuration card. There is no global mode and no silent fallback: the API
rejects OpenShell for another driver or worker.

**Status: proven end-to-end** on OpenShell 0.0.86 (macOS, Docker driver, host
gateway). A piped prompt reaches `claude` in the sandbox and stream-json flows
back, authenticated on the **Max/Pro subscription** (`apiKeySource=none`), with
`api.anthropic.com` allowed by our policy. See "Remaining" at the bottom for the
the orchestrator workspace round-tripping correctly.

## One-time host setup

The gateway + Docker compute driver run on the **host** (the containerized
scheduler points at them). Chosen deployment: host gateway + Docker driver.

```sh
# 1. Install OpenShell (official NVIDIA installer; starts the local gateway on :17670)
curl -LsSf https://raw.githubusercontent.com/NVIDIA/OpenShell/main/install.sh | sh
openshell status          # -> Connected, https://127.0.0.1:17670

# 2. BYOC sandbox image (has claude + iproute2 + the high-uid `sandbox` user).
#    We build our own to avoid the gated ghcr.io/nvidia/openshell-community images.
docker build -t zevo-openshell-sandbox:latest ops/openshell/sandbox

# 3. Provider carrying the Max OAuth token. The built-in `claude-code` provider
#    type satisfies OpenShell's agent binding AND (created this way) injects the
#    token as $CLAUDE_CODE_OAUTH_TOKEN inside the sandbox — subscription auth,
#    no metered API key.
export CLAUDE_CODE_OAUTH_TOKEN="$(grep '^CLAUDE_CODE_OAUTH_TOKEN=' .env | cut -d= -f2-)"
openshell provider create --name claude-code --type claude-code \
  --credential CLAUDE_CODE_OAUTH_TOKEN
```

`claude-code-oauth.profile.yaml` here is an *alternative* custom provider profile
(inject OAuth under a custom type) kept for reference / BYOC-without-agent-binding
setups; the built-in `claude-code` provider above is the simpler proven path.

## Turn it on

```sh
# .env — containers reach the host gateway here (compose already defaults these):
ZEVO_OPENSHELL_GATEWAY=https://host.docker.internal:17670
ZEVO_OPENSHELL_GATEWAY_INSECURE=1
```
```sh
docker compose up -d --force-recreate backend scheduler holdout-scheduler
```

Then open **Agents → orchestrator → Configuration**, choose `claude_cli`, choose
`OpenShell sandbox`, and save.

The driver then launches the orchestrator as:
```
openshell sandbox create --no-keep --no-tty \
  --from zevo-openshell-sandbox:latest --provider claude-code \
  --policy /app/ops/openshell/zevo.yaml --name zevo-<agent>-<ticket> -- \
  claude -p --output-format stream-json --verbose … -
```

## Verify by hand

```sh
printf 'Reply with exactly one word: PONG' | \
  openshell sandbox create --no-keep --no-tty \
    --from zevo-openshell-sandbox:latest --provider claude-code \
    --policy ops/openshell/zevo.yaml -- \
    claude -p --output-format stream-json --verbose -
# expect a final result line: "is_error":false,"result":"PONG"
```

## Policy notes (learned the hard way — see zevo.yaml)

- **Every network endpoint needs a `binaries` allowlist.** Without it OpenShell
  denies the connection (`CONNECT api.anthropic.com:443 not permitted by policy`).
  claude calls out via `node`; data/train skills via `python3`/`pip`/`curl`.
- **Protocols:** `rest | websocket | graphql | sql | json-rpc | mcp` — there is no
  raw-TCP tunnel; OpenShell L7-proxies (TLS-terminating) and the agent trusts its CA.
- **No `host: "*"` wildcards** — a middle-label wildcard must be a whole label.
- The sandbox user's HOME is `/home/sandbox` (not `/root`); keep it `read_write`.

## Workspace I/O

The driver now drives a full lifecycle around each sandboxed heartbeat
(`src/zevo/engine/agent/drivers/sandbox.py`):

1. `open_sandbox()` — create a keepalive sandbox (poll until **Ready**) and
   `sandbox upload` the ticket workspace into `/sandbox/<leaf>`;
2. `sandbox_exec_argv()` — run the agent via `sandbox exec --workdir /sandbox/<leaf>`
   (stdin prompt + stream-json stdout proxy through, as proven);
3. `close_sandbox()` — `sandbox download` the workspace back (merging artifacts
   the agent wrote) then `sandbox delete`.

Live-validated: a command that read an uploaded input and wrote `out/result.txt`
inside the sandbox produced that file back on the host, with clean teardown.

### Why workers are deliberately excluded
Only the ticket's OWN workspace dir round-trips. Agents that read/write ABSOLUTE
host paths outside it (cross-stage refs, `/app/data`, the run root) or SSH to
GPU boxes need those paths inside the sandbox **and** opened in the policy. SSH
egress to a rented GPU box also needs a per-run policy naming the specific host
(no static `host: "*"` allow-all). The current API and UI therefore permit this
sandbox only for the orchestrator; they do not present a switch that workers
would silently ignore or that would fail later during training.
