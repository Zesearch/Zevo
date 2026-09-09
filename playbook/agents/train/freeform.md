Input is the shared `FreeformInput`: `ticket_id`, `agent_id`, `request`,
`attachments`, `work_dir`, and `run_id`. There is no separate hints object.

The request must identify, or the attachments must unambiguously establish, a
prepared training dataset and `device_info.json`. Training remains remote GPU
work. Resolve `base_model`, `training_method`, the independent prompt contract,
the loss contract, `generation_backend`, and hyperparameters from explicit
request text first, then from verified artifact evidence and the selected Train
Skill. Record all resolutions in `Starting:` and `__CONFIG__`.

Choose exactly one installed Train Skill. Do not force SFT onto preference or
reward data, invent a missing teacher/reward, replace the provided dataset, or
fall back to CPU. Ambiguous required inputs produce a `failed` `TrainResult`.
