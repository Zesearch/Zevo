# BYOC (bring-your-own-container) sandbox image for ZEVO agents running under
# NVIDIA OpenShell. Avoids the gated ghcr.io/nvidia/openshell-community images
# and lets us attach the built-in `claude-code` provider carrying our Max/Pro
# CLAUDE_CODE_OAUTH_TOKEN, so the agent runs on the subscription (no API key).
#
# Per OpenShell BYOC requirements:
#   - standard Linux base (no distroless/scratch);
#   - install iproute2 so the supervisor can set up the network namespace;
#   - a non-root `sandbox` user at a HIGH uid/gid (1000660000) to avoid host
#     uid collisions when running without userns remapping;
#   - /sandbox writable by that user.
# The image CMD/ENTRYPOINT is replaced at runtime by the OpenShell supervisor;
# the agent command is supplied via `openshell sandbox create … -- claude …`.
#
# Build+run via: openshell sandbox create --from ops/openshell/sandbox …
# (or prebuild: docker build -t zevo-openshell-sandbox:latest ops/openshell/sandbox)
FROM node:20-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git openssh-client sshpass rsync python3 python3-pip jq \
        iproute2 \
    && rm -rf /var/lib/apt/lists/*

# The agent CLI the driver invokes inside the sandbox (npm installs it onto
# PATH at /usr/local/bin/claude).
# Same pin as Dockerfile.backend / Dockerfile.scheduler so sandboxed and
# unsandboxed agents run the same CLI.
ARG CLAUDE_CODE_VERSION=2.1.232
RUN npm install -g "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" && claude --version

# Non-root sandbox user at the high uid/gid OpenShell expects, owning /sandbox.
RUN groupadd -g 1000660000 sandbox \
    && useradd -m -u 1000660000 -g 1000660000 -s /bin/bash sandbox \
    && mkdir -p /sandbox && chown -R sandbox:sandbox /sandbox

WORKDIR /sandbox
