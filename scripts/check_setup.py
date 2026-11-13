"""Check package/source/asset availability without loading 30 GB of model weights."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'vendor/ThinkMorph')
    p.add_argument('--model-dir',type=Path,default=ROOT/'models/ThinkMorph-7B')
    p.add_argument('--gpu',action='store_true');p.add_argument('--eval',action='store_true');a=p.parse_args()
    os.environ.setdefault('LMUData',str(ROOT/'data/benchmarks'))
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'cache/matplotlib'))
    Path(os.environ['LMUData']).mkdir(parents=True,exist_ok=True)
    assert sys.version_info[:2]==(3,10),'Use the validated Python 3.10 environment'
    packages=['torch','torchvision','transformers','peft','accelerate','flash_attn','pyarrow','Pillow','numpy','wandb']
    print(json.dumps({k:importlib.metadata.version(k) for k in packages},indent=2))
    sys.path.insert(0,str(a.source.resolve()))
    import torch, flash_attn
    import torchvision
    import transformers, peft
    from modeling.bagel import Bagel
    from inferencer import InterleaveInferencer
    from data.transforms import ImageTransform
    expected=json.loads((ROOT/'provenance/upstream.json').read_text())['ThinkMorph']['revision']
    actual=subprocess.check_output(['git','-C',str(a.source),'rev-parse','HEAD'],text=True).strip()
    assert actual==expected,'Native upstream revision mismatch'
    manifest=json.loads((ROOT/'provenance/model.json').read_text())
    for item in manifest['files']:
        path=a.model_dir/item['file'];assert path.is_file() and path.stat().st_size==item['bytes'],str(path)
    snapshots=json.loads((ROOT/'provenance/source_snapshots.json').read_text())
    # Snapshot layout is intentionally explicit so profile drift is visible.
    entries=snapshots['files'] if isinstance(snapshots,dict) and 'files' in snapshots else snapshots
    for name,item in entries.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==item['sha256'],name
    if a.eval:
        sys.path.insert(0,str(ROOT/'vendor/VLMEvalKit'))
        from vlmeval.dataset import build_dataset
        from vlmeval.vlm.ThinkMorph import ThinkMorph
    if a.gpu:
        assert torch.cuda.is_available(),'No CUDA GPU visible'
        print(json.dumps([dict(index=i,name=torch.cuda.get_device_name(i),vram_gib=torch.cuda.get_device_properties(i).total_memory/2**30) for i in range(torch.cuda.device_count())],indent=2))
    print('Setup imports, source hashes and model file sizes passed. Use download_assets.py --verify-only for full model hashes.')

if __name__=='__main__':main()
