<div align="center">

<img src="figs/title.svg" alt="Zevo: from Zero to evolved — A Self-Improving System For Evolving Language Models" width="100%">

[![Website](https://img.shields.io/badge/Website-zevoai.dev-E94F17?style=for-the-badge)](https://zevoai.dev)
[![User Manual](https://img.shields.io/badge/User%20Manual-Read%20the%20Guide-2563EB?style=for-the-badge)](https://zevoai.dev/docs)
[![License](https://img.shields.io/badge/License-Apache%202.0-1F2937?style=for-the-badge)](LICENSE)

<h3>
  <a href="mailto:haoyan.yang@stonybrook.edu">Haoyan Yang</a> ·
  <a href="mailto:saiakhilkogilathota@gmail.com">Sai Akhil Kogilathota</a> ·
  <a href="mailto:jiawei.zhou.1@stonybrook.edu">Jiawei Zhou</a>
  <br><br>
  Zesearch NLP Lab, Stony Brook University
</h3>

</div>

[![Zevo introduction](figs/intro.png)](https://zevoai.dev)

## News

- **09/2026** — Open-sourced the preview version of Zevo and published the [User Manual](https://zevoai.dev/docs).
- **08/2026** — Released the [Zevo project website](https://zevoai.dev).

## Overview

Zevo is a self-improving system that uses multi-agent collaboration to autonomously evolve language models toward a user-defined objective.

![Human-led model training compared with Zevo](figs/human-vs-zevo.png)

- 🤖 **Autonomous**
  - Zevo plans and runs the complete improvement loop end to end without human intervention after launch.
- 🎯 **Reliable Evaluation**
  - Zevo supports user-defined test sets and metrics, so improvement is measured against each user's objective.
  - Zevo optimizes on validation and keeps the optimization loop schema-isolated from the test set, preventing test-set overfitting.
- ⚖️ **Fair Comparison**
  - Zevo fixes prompts, chat templates, decoding strategies, and inference settings in the baseline round so later score gains reflect model evolution rather than a moving setup.
- 🧠 **Flexible Agent Stack**
  - Zevo supports many model families as agents, including GPT models from OpenAI, Claude models from Anthropic, and the latest open-source models through Amazon Bedrock and OpenRouter.
- ⚡ **Infrastructure-Agnostic**
  - Zevo supports rented cloud GPUs, user-owned Slurm clusters, and dedicated GPU machines.
  - Zevo automatically queues and runs jobs, monitors their status, and reads back the results.
- 🔒 **Secure**
  - Zevo keeps secrets and private run state in the local deployment and provides sandbox mode with explicitly scoped agent permissions.
- 🔍 **Transparent**
  - Zevo records every decision, configuration, dataset, model, prediction, score, log, and artifact.
- ♻️ **Resilient**
  - Zevo uses schema, artifact, and heuristic checks to detect failures, automatically repairs recoverable issues, and safely resumes the workflow.

## Agentic Workflow

![Zevo agentic architecture](figs/architecture.gif)

The **Orchestrator** is Zevo's brain: it directs the experiment, interprets results, selects the next direction, and works with each specialized agent through structured turns. Specialized agents do not communicate directly with one another.

The specialized agents include:

- **Infrastructure** — Manages GPU execution, including resource provisioning, job submission, queue monitoring, and cleanup.
- **Data** — Acquires and prepares training and validation data, and creates a test set when needed.
- **Train** — Trains models on the prepared data using either a user-specified or Zevo-selected method.
- **Inference** — Generates structured predictions using the trained model.
- **Evaluation** — Measures validation and held-out test performance deterministically from the generated predictions.
- **Registry** — Records the champion model selected by validation performance and preserves its lineage.

## Level of Control

Both dimensions describe how much control the user delegates to Zevo, but at different layers. System control modes govern workflow-level control, including evaluation setup, agent configuration, and whether Zevo runs the complete loop. Training autonomy levels govern optimization-level control, including whether Zevo can choose the training data, model, and method. Any end-to-end mode can be paired with L1 to L4.

### System Control Modes

**From M1 to M4, higher modes delegate more workflow-level control to Zevo.**

| Mode | Zevo Runs End-to-End Workflow | Zevo Decides Test Set | No User-Specific Constraints on Zevo Agents |
|---|:---:|:---:|:---:|
| **M1 · Single Stage**<br>The user sends one focused task to one specialized agent. | ✕ | ✕ | ✕ |
| **M2 · Customized**<br>The user provides the evaluation and adds constraints to selected agents. Zevo runs the complete workflow. | ✓ | ✕ | ✕ |
| **M3 · Standard**<br>The user provides the evaluation. Zevo runs the complete workflow. | ✓ | ✕ | ✓ |
| **M4 · Auto**<br>Zevo creates the evaluation and runs the complete workflow. | ✓ | ✓ | ✓ |

### Training Autonomy Levels

**From L1 to L4, higher levels delegate more optimization-level control to Zevo.**

<table width="100%">
  <thead>
    <tr>
      <th align="left" width="55%">Level</th>
      <th align="center" width="15%">Training Data</th>
      <th align="center" width="15%">Training Model</th>
      <th align="center" width="15%">Training Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>L1 · Entry Autonomous</strong><br>Zevo iterates within the given data, model, and method.</td>
      <td align="center"><strong>User</strong></td>
      <td align="center"><strong>User</strong></td>
      <td align="center"><strong>User</strong></td>
    </tr>
    <tr>
      <td><strong>L2 · Constraint Autonomous</strong><br>Zevo collects and even creates the training data.</td>
      <td align="center"><strong>Zevo</strong></td>
      <td align="center"><strong>User</strong></td>
      <td align="center"><strong>User</strong></td>
    </tr>
    <tr>
      <td><strong>L3 · Partially Autonomous</strong><br>Zevo picks the training method as well.</td>
      <td align="center"><strong>Zevo</strong></td>
      <td align="center"><strong>User</strong></td>
      <td align="center"><strong>Zevo</strong></td>
    </tr>
    <tr>
      <td><strong>L4 · Fully Autonomous</strong><br>Zevo decides everything from the objective alone.</td>
      <td align="center"><strong>Zevo</strong></td>
      <td align="center"><strong>Zevo</strong></td>
      <td align="center"><strong>Zevo</strong></td>
    </tr>
  </tbody>
</table>

## Interfaces

Zevo supports both a CLI and Web UI for launching, monitoring, and inspecting the complete improvement loop.

<p align="center">
  <img src="figs/cli.png" alt="Zevo CLI" width="47%">
  &nbsp;&nbsp;&nbsp;
  <img src="figs/ui.png" alt="Zevo Web UI" width="47%">
  <br><br>
  <strong>Zevo CLI</strong>
  &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
  <strong>Zevo Web UI</strong>
</p>

## Citation

If you use Zevo in your research, please cite:

```bibtex
@software{zevo2026,
  author = {Haoyan Yang and Sai Akhil Kogilathota and Jiawei Zhou},
  title  = {Zevo: A Self-Improving System for Evolving Language Models},
  year   = {2026},
  url    = {https://github.com/Zesearch/Zevo-ZeroToEvolved}
}
```

Zevo is released under the [Apache License 2.0](LICENSE).
