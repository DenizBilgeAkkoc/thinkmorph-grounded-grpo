"""Export a committed checkpoint's adapters on CPU; never touch the live policy."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import torch


def digest_adapters(adapters):
    digest = hashlib.sha256()
    for name, value in adapters.items():
        assert 'lora_' in name and value.device.type == 'cpu'
        assert torch.isfinite(value).all(), name
        tensor = value.contiguous()
        digest.update(f'{name}:{tuple(tensor.shape)}:{tensor.dtype}'.encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-digest', required=True)
    parser.add_argument('--expected-update', type=int, required=True)
    parser.add_argument('--expected-questions', type=int, required=True)
    args = parser.parse_args()
    from train import committed_checkpoint
    marker = committed_checkpoint(args.checkpoint)
    assert marker['policy_digest'] == args.expected_digest
    assert marker['optimizer_updates'] == args.expected_update and marker['group'] + 1 == args.expected_questions
    torch.set_num_threads(2)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    actual = digest_adapters(state['adapters'])
    assert actual == args.expected_digest == state['policy_digest']
    assert state['optimizer_updates'] == args.expected_update
    assert state['group'] + 1 == args.expected_questions
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / 'adapters.pt'
    assert not target.exists(), 'Export already exists; do not overwrite it'
    payload = dict(schema_version=1, adapters=state['adapters'], policy_digest=actual,
                   optimizer_updates=state['optimizer_updates'], group=state['group'],
                   lora=dict(rank=64, alpha=128))
    temporary = target.with_suffix('.pt.tmp')
    torch.save(payload, temporary)
    temporary.replace(target)
    sha = hashlib.sha256()
    with target.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            sha.update(block)
    metadata = dict(schema_version=1, exported_at_utc=datetime.now(timezone.utc).isoformat(),
        source_checkpoint=str(args.checkpoint), policy_digest=actual,
        optimizer_updates=state['optimizer_updates'], processed_questions=state['group'] + 1,
        file=target.name, bytes=target.stat().st_size, sha256=sha.hexdigest(),
        adapter_tensor_count=len(state['adapters']),
        adapter_parameter_count=sum(t.numel() for t in state['adapters'].values()),
        lora=payload['lora'], cpu_only=True, optimizer_excluded=True,
        training_configuration=state['configuration'])
    (args.output / 'Export_Manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({k: v for k, v in metadata.items() if k != 'training_configuration'}), flush=True)


if __name__ == '__main__':
    main()
