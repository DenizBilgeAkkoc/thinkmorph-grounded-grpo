"""Download pinned upstream source and native model assets; never run training."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
SOURCES={
    'ThinkMorph':('https://github.com/ThinkMorph/ThinkMorph.git','c1a48adfa212259c8ad79dfd9d05d87c27340cef'),
    'VLMEvalKit':('https://github.com/hychaochao/VLMEvalKit_Thinkmorph.git','cddb6a38ac5dfaf64020d6aa3964506e26006568'),
}

def checkout(name):
    url,revision=SOURCES[name];path=ROOT/'vendor'/name;path.parent.mkdir(exist_ok=True)
    if not path.exists():
        subprocess.run(['git','init',str(path)],check=True)
        subprocess.run(['git','-C',str(path),'remote','add','origin',url],check=True)
        subprocess.run(['git','-C',str(path),'fetch','--depth=1','origin',revision],check=True)
        subprocess.run(['git','-C',str(path),'checkout','--detach',revision],check=True)
    actual=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
    if actual!=revision:raise RuntimeError(f'{name} checkout differs from pinned revision; use a clean vendor directory')
    patch=ROOT/'patches/vlmeval.patch'
    if name=='VLMEvalKit':
        reverse=subprocess.run(['git','-C',str(path),'apply','--reverse','--check',str(patch)],capture_output=True)
        if reverse.returncode:
            subprocess.run(['git','-C',str(path),'apply','--check',str(patch)],check=True)
            subprocess.run(['git','-C',str(path),'apply',str(patch)],check=True)
    elif subprocess.check_output(['git','-C',str(path),'status','--porcelain','--untracked-files=no'],text=True).strip():
        raise RuntimeError('Native ThinkMorph source has modifications')
    print(f'{name}: {revision}',flush=True)

def verify(path,item):
    size=path.stat().st_size
    if size!=item['bytes']:raise ValueError('Model file size mismatch: '+str(path))
    h=hashlib.new(item['algorithm'])
    # HF small-file SHA1s are Git blob IDs, not raw file SHA1s.
    if item['algorithm']=='sha1':h.update(f'blob {size}\0'.encode())
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    if h.hexdigest()!=item['digest']:raise ValueError('Model checksum mismatch: '+str(path))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=Path,default=ROOT/'models/ThinkMorph-7B')
    p.add_argument('--source-only',action='store_true');p.add_argument('--eval',action='store_true')
    p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    if not a.verify_only:
        checkout('ThinkMorph')
        if a.eval:checkout('VLMEvalKit')
    if a.source_only:return
    manifest=json.loads((ROOT/'provenance/model.json').read_text())
    if not a.verify_only:
        from huggingface_hub import snapshot_download
        snapshot_download(manifest['repository'],revision=manifest['revision'],local_dir=str(a.model_dir),
                          allow_patterns=[v['file'] for v in manifest['files']])
    for item in manifest['files']:
        verify(a.model_dir/item['file'],item);print('Verified '+item['file'],flush=True)

if __name__=='__main__':main()
