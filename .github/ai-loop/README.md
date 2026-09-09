# AI issue-to-PR loop

Three Claude Max accounts run a closed loop on this repo. An issue becomes a
draft PR, a second account reviews it, the first account answers the review, and
the cycle repeats until the reviewer is satisfied or the round cap is hit.

## The loop

```
issue opened (by someone with write access)
        │
        ▼
  ai-implement ──► draft PR on branch ai/issue-<n>, label ai:round-1
        │
        ▼
   ai-review  ──► inline comments + verdict label
        │
        ├── ai:approved          ──► stop. A human marks the PR ready and merges.
        ├── ai:changes-requested ──► ai-address ──► push fixes, round+1, back to ai-review
        └── no verdict           ──► ai:needs-human, stop.

  round > 3 at any point         ──► ai:needs-human, stop.
```

## Why the stages dispatch each other explicitly

A PR opened by a workflow using the default `GITHUB_TOKEN` does **not** fire
`pull_request: opened`. That is GitHub's guard against recursive workflow runs,
and it silently breaks the obvious version of this design: the reviewer never
wakes up. `workflow_dispatch` and `repository_dispatch` are the documented
exceptions, so every stage ends by calling `gh workflow run` on the next one.

Two consequences. All three `ai-*.yml` files must exist on the **default
branch** or the dispatch has nothing to target. And no PAT or GitHub App is
needed to trigger anything, so the chain works without org-owner involvement.

Separately, `claude-code-action` exchanges for a Claude GitHub App token unless
you hand it one. The App is not installed on this org, so every step passes
`github_token: ${{ github.token }}` and the agents act as `github-actions[bot]`.
Installing https://github.com/apps/claude and dropping that line is the
alternative; it only changes which identity the commits and comments carry.

## State lives in labels

The PR's position in the loop is a label, not something the agent remembers.
`ai:round-<n>` is the counter, `ai:approved` / `ai:changes-requested` is the
verdict. The routing step reads labels with `gh` and decides in bash. The model
supplies the judgment; the script owns the control flow. If the reviewer fails
to record a verdict, the router stops the loop rather than guessing.

## Accounts

| Secret | Account | Role |
| --- | --- | --- |
| `CLAUDE_OAUTH_IMPLEMENTER` | A | Writes the code, answers review comments |
| `CLAUDE_OAUTH_REVIEWER` | B | Reviews only. Runs with `contents: read`, so it cannot push. |
| `CLAUDE_OAUTH_FAILOVER` | C | Retries a stage whose primary account failed, usually a 5-hour rate limit |

Generate each token while logged into that account: `claude setup-token`.

### Telling them apart on a PR

All three post as `github-actions[bot]`, because that is whose token the action
is given. GitHub shows one name and one avatar for the reviewer, the
implementer's replies inside the reviewer's own threads, and the workflow's own
routing messages. The action's `bot_name` and `bot_id` inputs do not help; they
set git commit authorship, not comment authorship.

So each voice labels itself on its first line:

| Line | Who |
| --- | --- |
| `🔨 **Implementer**` | account A, in the PR body and in replies to review threads |
| `🔍 **Reviewer**` | account B, on every inline comment and the summary |
| `⚙️ **Loop**` | the workflow's own bash, not an agent |

### Giving them separate identities

The workflows already mint a per-stage GitHub App token and hand it to the
action. Until the App secrets exist each stage falls back to `GITHUB_TOKEN`, so
the loop runs either way; setting the secrets is what switches the identities on.

Register two Apps at `https://github.com/organizations/<org>/settings/apps/new`,
one per role. Both need **Metadata: read** plus:

| App | Repository permissions |
| --- | --- |
| implementer | Contents: read & write, Issues: read & write, Pull requests: read & write |
| reviewer | Contents: read, Issues: read & write, Pull requests: read & write |

The reviewer gets Contents read-only on purpose. It has no reason to push, and
the token is the thing enforcing that rather than the prompt.

Install both on the org, granting access to the repos that run the loop.
Generate a private key for each, then set four secrets per repo:

```sh
gh secret set AI_IMPLEMENTER_APP_CLIENT_ID  -R <owner>/<repo> --body 'Iv1....'
gh secret set AI_IMPLEMENTER_APP_PRIVATE_KEY -R <owner>/<repo> < implementer.private-key.pem
gh secret set AI_REVIEWER_APP_CLIENT_ID     -R <owner>/<repo> --body 'Iv1....'
gh secret set AI_REVIEWER_APP_PRIVATE_KEY   -R <owner>/<repo> < reviewer.private-key.pem
```

Use the App's **Client ID** (`Iv1...`), not the numeric App ID; `app-id` is
deprecated in `create-github-app-token`.

Two things do not change. `github.actor` on a chained stage stays
`github-actions[bot]` regardless of which token the action holds, because the
actor is whoever triggered the run, so `allowed_bots` is still required. And the
routing bash keeps using `GITHUB_TOKEN`, which is deliberate: `gh workflow run`
under `GITHUB_TOKEN` is the documented exception that makes the chain fire, and
it keeps the loop's own messages visually distinct from both agents.

Once both Apps are live the `🔨` and `🔍` headers are redundant, but harmless.

## Cost

Draft PRs are used deliberately. The expensive CI workflow skips drafts, so the
review rounds do not each trigger a Docker build and a Terraform plan. CI runs
once, when a human marks the PR ready for review.

## Changing behaviour

Edit the prompts in `prompts/`, not the YAML. Each workflow loads its prompt
through `envsubst`, so `$REPO`, `$PR`, `$ISSUE`, `$BRANCH`, `$ROUND` and
`$MAX_ROUNDS` interpolate. Raise or lower the round cap with `MAX_ROUNDS` in
`ai-review.yml` and `ai-address.yml`.

## Operating it

- Opt an issue out: add `ai:skip` before opening, or just close the PR.
- Re-run a stage by hand: Actions tab, `ai-implement` / `ai-review` /
  `ai-address`, "Run workflow", pass the issue or PR number.
- Stop a runaway loop: remove the `ai:round-*` label, or disable the workflows.

## Setup

Three prerequisites, in the order they bite.

**1. Let Actions open pull requests.** Settings, Actions, General, Workflow
permissions, "Allow GitHub Actions to create and approve pull requests". Without
it `ai-implement` does the whole job and then fails on the last step with
`GitHub Actions is not permitted to create or approve pull requests
(createPullRequest)`, leaving a pushed branch and no PR. This is a repository
setting, not a workflow permission, so granting `pull-requests: write` does not
substitute for it. An organisation-level policy overrides the repository
setting, so on an org repo an **owner** has to enable it at
`https://github.com/organizations/<org>/settings/actions`; a repo admin cannot.

The loop never uses the "approve" half. The reviewer records its verdict as a
label precisely so no agent ever calls the approval API.

**2. Put the workflows on the default branch.** Each stage starts the next with
`gh workflow run`, which can only target a workflow that exists on the default
branch. Nothing chains until these files are merged.

**3. Set the secrets.**

```sh
export CLAUDE_OAUTH_IMPLEMENTER=... CLAUDE_OAUTH_REVIEWER=... CLAUDE_OAUTH_FAILOVER=...
./bootstrap.sh <owner>/<repo>
```
