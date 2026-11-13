"""Seeded MMVP/VStar inference and exact scoring from a verified LoRA export."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[1]

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)

def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()

def install_adapters(model, protocol):
    import torch
    from peft import LoraConfig, inject_adapter_in_model
    from export_adapters import digest_adapters

    path = Path(protocol['adapter_file'])
    assert file_digest(path) == protocol['expected_adapter_sha256'], 'Adapter file checksum mismatch'
    state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    assert state['optimizer_updates'] == protocol['checkpoint_updates']
    assert state['group'] + 1 == protocol['checkpoint_processed_questions']
    assert state['lora'] == protocol['lora']
    assert digest_adapters(state['adapters']) == state['policy_digest'] == protocol['expected_policy_digest']
    cfg = LoraConfig(r=state['lora']['rank'], lora_alpha=state['lora']['alpha'],
        lora_dropout=0.0, bias='none',
        target_modules=['self_attn.q_proj_moe_gen', 'self_attn.k_proj_moe_gen',
            'self_attn.v_proj_moe_gen', 'self_attn.o_proj_moe_gen',
            'mlp_moe_gen.gate_proj', 'mlp_moe_gen.up_proj', 'mlp_moe_gen.down_proj',
            'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
            'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'])
    inject_adapter_in_model(cfg, model.language_model)
    actual = {name: p for name, p in model.named_parameters() if 'lora_' in name}
    assert set(actual) == set(state['adapters']), 'Adapter key mismatch'
    with torch.no_grad():
        for name, parameter in actual.items():
            saved = state['adapters'][name]
            assert parameter.shape == saved.shape and saved.dtype == torch.float32, name
            parameter.data = parameter.data.float()
            parameter.copy_(saved.to(parameter.device))
    model.eval().requires_grad_(False)
    loaded = {name: value.detach().cpu() for name, value in actual.items()}
    assert digest_adapters(loaded) == state['policy_digest'], 'Loaded policy differs from export'
    from peft.tuners.tuners_utils import BaseTunerLayer
    layers = [m for m in model.modules() if isinstance(m, BaseTunerLayer)]
    assert len(layers) == len(actual) // 2 and all(not m.disable_adapters for m in layers)
    return dict(policy_digest=state['policy_digest'], adapter_tensors=len(actual),
                active_adapter_layers=len(layers), dtype='float32', merged=False)


def output_identity(raw):
    from PIL import Image
    images=[]
    for path in re.findall(r'\[Image:\s*([^\n]+?)\]',raw):
        with Image.open(path) as source:
            rgb=source.convert('RGB')
            images.append(hashlib.sha256(f'RGB:{rgb.width}x{rgb.height}\0'.encode()+rgb.tobytes()).hexdigest())
    text=re.sub(r'\[Image:[^\n]+?\]','[Image: GENERATED_IMAGE]',raw)
    return dict(text_sha256=hashlib.sha256(text.encode()).hexdigest(),image_pixel_sha256=images)


def score(dataset,folder,protocol,records):
    import pandas as pd
    output=dataset.data.iloc[:len(records)].drop(columns=['image'],errors='ignore').copy()
    output['prediction']=[r['prediction'] for r in records]
    output['thinking']=[r['thinking'] for r in records]
    path=folder/f"checkpoint_{protocol['dataset']}.xlsx"
    output.to_excel(path,index=False,engine='xlsxwriter')
    scores=dataset.evaluate(str(path),model='exact_matching',nproc=2)
    detail=pd.read_excel(path.with_name(path.stem+'_exact_matching_result.xlsx'),keep_default_na=False)
    assert len(detail)==len(records) and detail['index'].is_unique
    hit_by_index=dict(zip(detail['index'].astype(str),detail['hit'].astype(bool)))
    assert set(hit_by_index)=={r['index'] for r in records}
    # Pair in original dataset row order, not string-sorted index order.
    hits=[hit_by_index[r['index']] for r in records]
    result=dict(dataset=protocol['dataset'],questions=len(records),correct=sum(hits),
        accuracy_percent=100*sum(hits)/len(hits),native_scores=scores.to_dict('records'),
        missing_final_answers=sum(r['prediction']=='[NO_FINAL_ANSWER]' for r in records),
        complete=len(records)==protocol['expected_questions'],protocol_sha256=file_digest(folder/'protocol.json'))
    if protocol['dataset']=='MMVP' and len(hits)%2==0:
        result['paired_accuracy_percent']=100*sum(a and b for a,b in zip(hits[::2],hits[1::2]))/(len(hits)//2)
    atomic_json(folder/'Results.json',result)
    print(json.dumps(result,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark',choices=('MMVP','VStarBench'),required=True)
    p.add_argument('--adapters',type=Path,required=True,help='Directory containing adapters.pt and Export_Manifest.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,default=ROOT/'models/ThinkMorph-7B')
    p.add_argument('--vlmeval-source',type=Path,default=ROOT/'vendor/VLMEvalKit')
    p.add_argument('--dataset-cache',type=Path,default=ROOT/'data/benchmarks')
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--limit',type=int);p.add_argument('--seed-base',type=int,default=42)
    p.add_argument('--score-only',action='store_true');a=p.parse_args()
    if not 0<=a.shard<a.shards:p.error('Require 0 <= shard < shards')
    for key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE'):os.environ.pop(key,None)
    os.environ.update(LMUData=str(a.dataset_cache.resolve()),SPLIT_THINK='1',PRINT_VANILLA='1',VLMEVAL_FAIL_FAST='1')
    a.dataset_cache.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(a.vlmeval_source.resolve()))
    from vlmeval.dataset import build_dataset
    settings=json.loads((ROOT/'evaluation/settings.json').read_text())
    export=json.loads((a.adapters/'Export_Manifest.json').read_text())
    dataset=build_dataset(a.benchmark);expected=settings['expected_questions'][a.benchmark]
    assert len(dataset)==expected and dataset.data['index'].is_unique
    limit=a.limit if a.limit is not None else expected
    if not 1<=limit<=expected:p.error('Invalid example limit')
    protocol=dict(dataset=a.benchmark,expected_questions=expected,limit=limit,seed_base=a.seed_base,
        model={**settings['model'],'model_path':str(a.model_dir.resolve())},
        adapter_file=str((a.adapters/'adapters.pt').resolve()),expected_adapter_sha256=export['sha256'],
        checkpoint_updates=export['optimizer_updates'],checkpoint_processed_questions=export['processed_questions'],
        expected_policy_digest=export['policy_digest'],lora=export['lora'],
        source_sha256=file_digest(__file__),vlmeval_patch_sha256=file_digest(ROOT/'patches/vlmeval.patch'),
        dataset_tsv_sha256=file_digest(a.dataset_cache/f'{a.benchmark}.tsv'))
    folder=a.output.resolve();folder.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (folder/'protocol.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        path=folder/'protocol.json'
        if path.exists():assert json.loads(path.read_text())==protocol,'Protocol changed; choose a fresh output folder'
        else:atomic_json(path,protocol)
    protocol_hash=file_digest(folder/'protocol.json')
    def load_records():
        records=[json.loads((folder/'records'/f'{i:04d}.json').read_text()) for i in range(limit)]
        for i,r in enumerate(records):
            assert r['position']==i and r['protocol_sha256']==protocol_hash
            assert r['seed']==a.seed_base+i and r['index']==str(dataset.data.iloc[i]['index'])
        return records
    if a.score_only:score(dataset,folder,protocol,load_records());return
    import numpy as np
    import torch
    from vlmeval.vlm.ThinkMorph import ThinkMorph
    assert torch.cuda.device_count()==1,'Set CUDA_VISIBLE_DEVICES to one GPU per worker'
    torch.set_num_threads(3)
    worker=folder/'workers'/str(a.shard);worker.mkdir(parents=True,exist_ok=True)
    with (worker/'worker.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        model=ThinkMorph(**protocol['model'],save_dir=str(folder/'generated_images'))
        assert all(x.device.type=='cuda' for x in model.model.parameters())
        loaded=install_adapters(model.model,protocol)
        assert model.inferencer.model is model.model
        model.set_dump_image(dataset.dump_image)
        assert not model.use_custom_prompt(a.benchmark)
        import importlib.metadata
        atomic_json(worker/'Load_Validation.json',dict(**loaded,gpu=torch.cuda.get_device_name(0),
            packages={k:importlib.metadata.version(k) for k in ('torch','transformers','peft','accelerate')},
            inference_hyper=model.inference_hyper,protocol_sha256=protocol_hash))
        selected=list(range(a.shard,limit,a.shards))
        atomic_json(worker/'Assignment.json',dict(positions=selected,seeds=[a.seed_base+i for i in selected],protocol_sha256=protocol_hash))
        def generate(prompt,seed):
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            torch.cuda.reset_peak_memory_stats();started=time.monotonic()
            with torch.inference_mode():raw=model.generate(prompt,dataset=a.benchmark)
            torch.cuda.synchronize()
            assert isinstance(raw,str) and raw.strip() and 'Failed to obtain answer' not in raw
            return raw,time.monotonic()-started,torch.cuda.max_memory_allocated()/2**30
        for n,position in enumerate(selected):
            row=dataset.data.iloc[position];target=folder/'records'/f'{position:04d}.json';seed=a.seed_base+position
            prompt=dataset.build_prompt(row)
            if target.exists():
                record=json.loads(target.read_text())
                assert record['protocol_sha256']==protocol_hash and record['seed']==seed
                assert record['index']==str(row['index']) and record['prompt']==prompt
                if position==selected[0]:
                    proof=json.loads((worker/'Seed_Repeatability.json').read_text());assert proof['exact_match'] and proof['seed']==seed
            else:
                atomic_json(worker/'Status.json',dict(status='generating',position=position,completed=n,target=len(selected)))
                raw,seconds,peak=generate(prompt,seed);identity=output_identity(raw)
                if position==selected[0]:
                    repeated,repeat_seconds,_=generate(prompt,seed);other=output_identity(repeated)
                    proof=dict(seed=seed,original=identity,repeated=other,exact_match=identity==other,
                        overhead_seconds=repeat_seconds,scope='Same hardware/software; cross-hardware identity is not guaranteed')
                    atomic_json(worker/'Seed_Repeatability.json',proof)
                    assert proof['exact_match'],'Seed repeatability failed'
                answer,thinking=model.split_thinking(raw)
                record=dict(position=position,index=str(row['index']),question=row['question'],seed=seed,
                    prompt=prompt,prediction=answer,thinking=thinking,raw_output=raw,seconds=seconds,
                    generated_images=raw.count('[Image:'),peak_gpu_allocated_gib=peak,
                    output_identity=identity,protocol_sha256=protocol_hash)
                atomic_json(target,record)
            atomic_json(worker/'Status.json',dict(status='running',completed=n+1,target=len(selected),last_position=position))
        atomic_json(worker/'COMPLETE.json',dict(positions=selected,protocol_sha256=protocol_hash))
        atomic_json(worker/'Status.json',dict(status='complete',completed=len(selected),target=len(selected)))
    if a.shards==1:score(dataset,folder,protocol,load_records())


if __name__=='__main__':main()
