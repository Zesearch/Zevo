---
name: <site-slug>
description: "Operate Zevo on <site-name>. Use only when ssh_host matches <login-host> or the work order explicitly names <site-name>."
---

# <Site Name>

Copy this file to `<site-slug>/SKILL.md`, replace every placeholder, and keep
the resulting site directory local. Zevo automatically discovers directories
that contain a file named `SKILL.md`; this template filename is not loaded.

## Match the site

- List accepted login hosts or aliases.
- State whether this is a Slurm `cluster` or a fixed GPU `instance`.
- Describe explicit non-matches so rules from another site are never reused.

## Validate configuration

- Require the configured SSH host, port, user, authentication method, absolute
  remote directory, and environment activation command.
- For clusters, validate the partition, account, QOS, container image, shared
  storage, GPU limits, CPU/RAM bounds, and wall-time rules against live state.
- Never put credentials, private-key contents, tokens, usernames, or a person's
  concrete storage paths in this file.

## Discover resources

- List bounded, read-only commands for checking the host, scheduler, available
  GPU types, associations, quotas, and environment readiness.
- Treat live scheduler and device output as more authoritative than this Skill.
- Use low-frequency monitoring with backoff on shared schedulers.

## Execute

- For a Slurm cluster, require Train and Inference to write a finite
  `train.sbatch` or `predict.sbatch`, submit that file, register the job ID,
  and let the backend monitor it through queued, running, and terminal states.
- For a fixed instance, describe direct execution and idle-GPU selection.
- Define site-specific checkpoint/resume, cache, temporary-directory,
  container, mount, log, and cleanup rules without weakening Zevo ownership.
- Never use an empty allocation holder or `sleep infinity` as normal execution.

## Release and report

- Release or cancel only the exact Zevo-owned job or process.
- Record the selected resources, scheduler/job identity, observed device facts,
  output paths, and terminal evidence without copying secrets into artifacts.

## Diagnose failures

Map common SSH, environment, partition/QOS, GRES, queue, timeout, container,
cache, checkpoint, and output-validation failures to safe, bounded recovery
actions. Do not silently switch sites, accounts, partitions, or GPU types.
