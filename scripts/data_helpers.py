"""Export a fixed, pinned, train-only ThinkMorph pilot using parquet range reads.

Only pid, question, answer and original image columns are read. No generation,
judge calls, reference traces, held-out examples or remote SSH are involved.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import threading
import time

import pyarrow.parquet as pq
from PIL import Image
import requests

TASKS = ('Visual_Search','Chart_Refocus','Jigsaw_Assembly','Spatial_Navigation')
REVISIONS = dict(zip(TASKS,('f4e1839ca893a151355b7c843b737bf907cf0a05',
    'f088c9c5b1084e1287fd4f969e6e3bcbcfd0d0e6','9f437eb85a7b62cc65dadd930f6c0a665f0ea6b4',
    'b7567d9f2fcd096f5bf225cc4e64fc5c2b360897')))
COLUMNS = ['pid','question','answer','problem_image_0']


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')


def atomic_json(path,value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    temporary.replace(path)


def image_identity(raw):
    with Image.open(io.BytesIO(raw)) as source:
        image = source.convert('RGB')
    width,height = image.size
    digest = sha(f'RGB:{width}x{height}\0'.encode()+image.tobytes())
    return image,digest


def input_identity(question,pixel_digest):
    return sha(canonical(dict(question=' '.join(question.split()),rgb_pixel_sha256=pixel_digest)))


class Audit:
    def __init__(self,path):
        self.path,self.lock = path,threading.Lock()

    def __call__(self,event,**values):
        row = dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),event=event,**values)
        with self.lock:
            with self.path.open('a') as stream:
                stream.write(json.dumps(row,sort_keys=True)+'\n')
            if event not in ('range_read','range_cache_hit'):
                print(json.dumps(row,sort_keys=True),flush=True)


class RangeReader(io.RawIOBase):
    """Seekable immutable public parquet; require exact HTTP 206 byte ranges."""
    def __init__(self,task,revision,file,cache,audit):
        super().__init__()
        self.task,self.file,self.audit = task,file,audit
        self.size,self.position = int(file['size']),0
        self.url = f"https://huggingface.co/datasets/ThinkMorph/{task}/resolve/{revision}/{file['rfilename']}"
        self.cache = cache/task/Path(file['rfilename']).name
        self.cache.mkdir(parents=True,exist_ok=True)
        self.session = requests.Session()
        self.requests,self.transferred,self.cached_bytes = 0,0,0
        self.ranges = {}

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self,offset,whence=0):
        target = offset if whence==0 else self.position+offset if whence==1 else self.size+offset if whence==2 else -1
        if target<0 or target>self.size:
            raise ValueError('Seek outside immutable source object')
        self.position = target
        return target

    def read(self,count=-1):
        if count is None or count<0:
            count = self.size-self.position
        count = min(count,self.size-self.position)
        if count == 0:
            return b''
        start,end = self.position,self.position+count-1
        cache_key = f'{start:012d}_{count:012d}'
        payload,checksum = self.cache/(cache_key+'.bin'),self.cache/(cache_key+'.sha256')
        raw = None
        if payload.exists() and checksum.exists():
            candidate = payload.read_bytes()
            if len(candidate)==count and sha(candidate)==checksum.read_text().strip():
                raw = candidate
                self.cached_bytes += count
                self.audit('range_cache_hit',task=self.task,parquet=self.file['rfilename'],start=start,bytes=count)
        if raw is None:
            started = time.monotonic()
            for attempt in range(3):
                try:
                    with self.session.get(self.url,headers={'Range':f'bytes={start}-{end}'},
                                          timeout=(15,60),stream=True) as response:
                        if response.status_code != 206:
                            raise RuntimeError(f'Expected HTTP206, got {response.status_code}')
                        if response.headers.get('Content-Range') != f'bytes {start}-{end}/{self.size}':
                            raise RuntimeError('Server returned an unexpected byte range')
                        raw = response.raw.read(count+1)
                        if len(raw)!=count:
                            raise RuntimeError('Wrong range payload length')
                    break
                except Exception as error:
                    self.audit('range_retry',task=self.task,parquet=self.file['rfilename'],start=start,
                               bytes=count,attempt=attempt,error_type=type(error).__name__,error=str(error))
                    if attempt == 2:
                        raise
                    time.sleep(1+attempt)
            self.requests += 1
            self.transferred += len(raw)
            temporary = payload.with_suffix('.tmp')
            temporary.write_bytes(raw)
            temporary.replace(payload)
            checksum.write_text(sha(raw)+'\n')
            self.audit('range_read',task=self.task,parquet=self.file['rfilename'],start=start,
                       bytes=count,sha256=sha(raw),seconds=time.monotonic()-started)
        self.ranges[cache_key] = dict(offset=start,length=count,sha256=sha(raw))
        self.position += len(raw)
        return raw

    def close(self):
        self.session.close()
        super().close()
