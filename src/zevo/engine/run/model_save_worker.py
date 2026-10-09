"""Run on the checkpoint host. JSON request/response over stdin/stdout only."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


def manifest(root: Path) -> dict[str, dict]:
    if not root.is_dir():
        raise ValueError('Checkpoint directory does not exist')
    files = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('Checkpoint must not contain symlinks')
        if path.is_file() and not path.name.startswith('.zevo-'):
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            files[str(path.relative_to(root))] = {'size': path.stat().st_size, 'sha256': digest.hexdigest()}
    configs = {'config.json', 'adapter_config.json'}
    if not configs.intersection(files) or not any(
        (name.endswith('.safetensors') or name.endswith('.bin')) and info['size'] > 0
        for name, info in files.items()
    ):
        raise ValueError('Checkpoint has no model config or nonempty weights')
    for name in ('model.safetensors.index.json', 'pytorch_model.bin.index.json'):
        if name in files:
            index = json.loads((root / name).read_text())
            shards = set(index.get('weight_map', {}).values())
            if not shards or not shards.issubset(files) or any(files[s]['size'] <= 0 for s in shards):
                raise ValueError('Checkpoint shard index is incomplete')
    return files


def save(request: dict) -> dict:
    source = Path(request['source']).resolve(strict=True)
    before = manifest(source)
    policy = request['policy']
    if policy['weights'] == 'remote':
        parent = Path(policy['remote_dir'])
        if not parent.is_absolute() or '..' in parent.parts or str(parent) == '/':
            raise ValueError('GPU destination must be an absolute directory other than /')
        parent.mkdir(parents=True, exist_ok=True)
        dest = parent.resolve() / request['tag']
        # Never replace an existing directory with a different checkpoint.
        if dest.is_symlink():
            raise ValueError("Destination must not be a symlink")
        if dest.exists():
            if manifest(dest) != before:
                raise ValueError('Destination already contains a different model; choose another directory')
        else:
            if source == dest or source in dest.parents:
                raise ValueError('Destination must not be inside the source checkpoint')
            stage = Path(tempfile.mkdtemp(prefix=f'.{request["tag"]}.partial-', dir=parent))
            try:
                shutil.copytree(source, stage, dirs_exist_ok=True)
                if manifest(stage) != before or manifest(source) != before:
                    raise ValueError('Checkpoint changed or copied bytes do not match')
                # A competing save cannot replace a nonempty destination.
                os.rename(stage, dest)
            finally:
                if stage.exists():
                    shutil.rmtree(stage)
        return {'storage': 'remote', 'path': str(dest), 'revision': '', 'files': len(before)}
    if policy['weights'] != 'hf':
        raise ValueError('Unsupported model destination')
    from huggingface_hub import HfApi
    api = HfApi(token=request['token'])
    repo = policy['hf_repo_id']
    api.create_repo(repo_id=repo, private=policy['hf_private'], exist_ok=True)
    repo_info = api.model_info(repo)
    if bool(repo_info.private) != policy['hf_private']:
        raise ValueError('Existing HF repository visibility differs from the selected visibility')
    stale_weights = [f.rfilename for f in (repo_info.siblings or [])
                     if f.rfilename not in before and f.rfilename.endswith(('.safetensors', '.bin', '.index.json'))]
    if stale_weights:
        raise ValueError('HF repository contains other model weights; choose an empty repository')
    commit = api.upload_folder(repo_id=repo, folder_path=str(source), ignore_patterns=['.zevo-*'],
                               commit_message=f'Zevo {request["tag"]} selected champion')
    revision = commit.oid
    uploaded = api.model_info(repo, revision=revision, files_metadata=True)
    sizes = {f.rfilename: f.size for f in uploaded.siblings}
    if any(sizes.get(name) != item['size'] for name, item in before.items()):
        raise ValueError('HF commit does not contain every checkpoint file with the expected size')
    if manifest(source) != before:
        raise ValueError('Checkpoint changed while uploading')
    return {'storage': 'hf', 'path': f'https://huggingface.co/{repo}/tree/{revision}',
            'revision': revision, 'files': len(before)}


if __name__ == '__main__':
    try:
        result = save(json.load(sys.stdin))
        print('ZEVO_MODEL_SAVED=' + json.dumps(result), flush=True)
    except Exception as exc:
        # Provider exceptions can contain request/credential details. Keep those
        # out of shared subprocess logs; structured policy errors are safe.
        message = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        print('Model preservation failed: ' + message, file=sys.stderr)
        sys.exit(1)
