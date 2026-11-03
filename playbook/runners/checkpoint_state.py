"""Portable, read-only checkpoint inventory shared by workers and the backend.

Inventory validation is not a tensor restore. The next training worker must load
all optimizer state before it can claim successful continuation.
"""


def verify_resume_checkpoint(path, *, step, planned_steps, world_size=1):
    import json
    from pathlib import Path

    root = Path(path).resolve()

    def required(name):
        candidate = (root / name).resolve()
        if (not candidate.is_relative_to(root) or not candidate.is_file()
                or candidate.stat().st_size == 0):
            raise ValueError(f"missing or invalid full-state checkpoint file: {name}")
        return candidate

    if type(world_size) is not int or world_size < 1:
        raise ValueError("invalid checkpoint world size")
    state = json.loads(required('trainer_state.json').read_text())
    if state.get('global_step') != step or state.get('max_steps') != planned_steps:
        raise ValueError("segment changed the full training plan or saved the wrong step")
    deepspeed = (root / 'latest').exists() or bool(list(root.glob('global_step*')))
    fsdp = bool(list(root.glob('optimizer*/**/*.distcp')))
    if deepspeed:
        tag = required('latest').read_text().strip()
        if tag != f'global_step{step}':
            raise ValueError('DeepSpeed latest tag differs from checkpoint step')
        # Supported layout: data-parallel ZeRO-3, one optimizer/model partition
        # per rank, no tensor/pipeline parallelism. Fail closed for other layouts.
        expected = set()
        for rank in range(world_size):
            model = f'{tag}/zero_pp_rank_{rank}_mp_rank_00_model_states.pt'
            required(model)
            expected.add(Path(model).name)
            choices = [f'{tag}/{prefix}zero_pp_rank_{rank}_mp_rank_00_optim_states.pt'
                       for prefix in ('bf16_', '')]
            found = [name for name in choices if (root / name).exists()]
            if len(found) != 1:
                raise ValueError(f'missing or ambiguous DeepSpeed optimizer rank {rank}')
            required(found[0])
            expected.add(Path(found[0]).name)
        actual = {p.name for p in (root / tag).glob('*states.pt')}
        if actual != expected:
            raise ValueError('unsupported DeepSpeed partition layout or world size')
    else:
        if fsdp:
            for p in root.glob('optimizer*/**/*.distcp'):
                required(p.relative_to(root))
        else:
            required('optimizer.pt')
        required('scheduler.pt')
    rng = (['rng_state.pth'] if world_size == 1 else
           [f'rng_state_{rank}.pth' for rank in range(world_size)])
    for name in rng:
        try:
            required(name)
        except ValueError as exc:
            raise ValueError(f'missing or invalid per-rank RNG state: {name}') from exc
    if not deepspeed:
        single = next((name for name in ('model.safetensors', 'pytorch_model.bin',
                      'adapter_model.safetensors', 'adapter_model.bin')
                       if (root / name).is_file()), None)
        index = next((name for name in ('model.safetensors.index.json', 'pytorch_model.bin.index.json')
                      if (root / name).is_file()), None)
        shards = list(root.glob('pytorch_model*/**/*.distcp'))
        if single:
            required(single)
        elif index:
            names = set(json.loads(required(index).read_text()).get('weight_map', {}).values())
            if not names:
                raise ValueError('empty checkpoint weight index')
            for name in names:
                required(name)
        elif shards:
            for p in shards:
                required(p.relative_to(root))
        else:
            raise ValueError('resumable checkpoint is missing model weights')
    return state
