"""Check Git-tracked release files for accidental credentials and large artifacts."""
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
PATTERNS={
    'Google API key':r'AIza[0-9A-Za-z_-]{35}',
    'Hugging Face token':r'hf_[0-9A-Za-z]{25,}',
    'GitHub token':r'(?:gh[pousr]_[0-9A-Za-z]{30,}|github_pat_[0-9A-Za-z_]{40,})',
    'private key':r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'credential in URL':r'https?://[^\s/:@]+:[^\s/@]+@',
    'private deployment path':r'/(?:scratch/dakkoc25|home/dakkoc25|lambda/nfs/bagel)/',
}

def main():
    names=subprocess.check_output(['git','-C',str(ROOT),'ls-files','-z']).decode().split('\0')
    failures=[];count=0;total=0
    for name in filter(None,names):
        path=ROOT/name;count+=1;raw=path.read_bytes();total+=len(raw)
        if len(raw)>10*1024*1024:failures.append((name,'file larger than 10 MiB'))
        if path.suffix in ('.pt','.safetensors','.parquet','.pem','.key','.png','.jpg'):
            failures.append((name,'weight, dataset, credential or generated binary'))
        if name.split('/')[0] in ('data','models','runs','cache','secrets','vendor'):
            failures.append((name,'excluded artifact directory'))
        for label,pattern in PATTERNS.items():
            if re.search(pattern,raw.decode('utf-8')):failures.append((name,label))
    # Never print matching credential contents.
    if failures:raise SystemExit('\n'.join(name+': '+label for name,label in failures))
    print(f'Scanned {count} tracked text files, {total} bytes; no detected credentials/private paths/large binaries.')

if __name__=='__main__':main()
