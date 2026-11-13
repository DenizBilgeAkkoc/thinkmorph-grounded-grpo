"""Load the explicitly selected frozen implementation before CUDA imports."""
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
profile=os.environ.get('THINKMORPH_PROFILE','optimized')
if profile not in ('production','optimized'):raise ValueError('Unknown execution profile')
source=ROOT/('snapshots/production' if profile=='production' else 'src')
sys.path.insert(0,str(source))
if os.environ.get('THINKMORPH_JUDGE_HTTP2')=='1':
    from gemini_http2 import install
    install()
from thinkmorph_grpo.train_distributed import main

if __name__=='__main__':main()
