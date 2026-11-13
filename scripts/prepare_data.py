"""Reconstruct the exact fixed training inputs from pinned public parquet rows."""
import argparse
from collections import defaultdict
import io
import json
from pathlib import Path
import pyarrow.parquet as pq
from data_helpers import RangeReader,Audit,REVISIONS,COLUMNS,sha,canonical,image_identity,input_identity,atomic_json

ROOT=Path(__file__).resolve().parents[1]

def materialize(selection,output,cache):
    rows=json.loads(Path(selection).read_text())
    if len({r['example_id'] for r in rows})!=len(rows):raise ValueError('Duplicate selected sample')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    if (output/'manifest.json').exists():raise ValueError('Fixed manifest already exists; choose a new output directory')
    audit=Audit(output/'download_audit.jsonl');groups=defaultdict(list)
    for index,row in enumerate(rows):
        if row['split']!='train' or row['revision']!=REVISIONS[row['task']]:raise ValueError('Unpinned training source')
        groups[(row['task'],row['source_parquet'])].append((index,row))
    result=[None]*len(rows)
    for (task,filename),selected in groups.items():
        meta=json.loads((ROOT/'provenance'/f'{task}.json').read_text())
        assert meta['sha']==REVISIONS[task] and meta['id']=='ThinkMorph/'+task
        file=next(v for v in meta['siblings'] if v['rfilename']==filename)
        with RangeReader(task,REVISIONS[task],file,Path(cache),audit) as reader:
            parquet=pq.ParquetFile(reader)
            for rg in sorted({row['source_parquet_row_group'] for _,row in selected}):
                table=parquet.read_row_group(rg,columns=COLUMNS,use_threads=False)
                offset=sum(parquet.metadata.row_group(i).num_rows for i in range(rg))
                for index,expected in selected:
                    if expected['source_parquet_row_group']!=rg:continue
                    record=table.slice(expected['source_parquet_row_index']-offset,1).to_pylist()[0]
                    if record['pid']!=expected['pid']:raise ValueError('Pinned row PID mismatch')
                    raw=record['problem_image_0']['bytes'];image,pixel_hash=image_identity(raw)
                    spec=expected['problem_image']
                    if sha(raw)!=spec['source_encoded_sha256'] or pixel_hash!=spec['pixel_sha256']:raise ValueError('Input image identity changed')
                    if input_identity(record['question'],pixel_hash)!=expected['input_sha256']:raise ValueError('Input question identity changed')
                    content=sha(canonical(dict(task=task,question=record['question'],answer=record['answer'],rgb_pixel_sha256=pixel_hash)))
                    if content!=expected['content_sha256']:raise ValueError('Question/answer/image content changed')
                    buf=io.BytesIO();image.save(buf,format='PNG',compress_level=6);encoded=buf.getvalue()
                    if sha(encoded)!=spec['sha256']:raise ValueError('PNG encoding differs: use the pinned Pillow build; never silently rebind historical hashes')
                    target=output/spec['relative_path'];target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(encoded)
                    result[index]={**expected,'question':record['question'],'answer':record['answer'],
                        'problem_image':{**spec,'path':str(target.resolve())}}
                del table
    assert all(result)
    atomic_json(output/'manifest.json',result)
    atomic_json(output/'Data_Receipt.json',dict(selection_sha256=sha(Path(selection).read_bytes()),
        questions=len(result),manifest_sha256=sha((output/'manifest.json').read_bytes()),
        columns=COLUMNS,pinned_revisions=REVISIONS,reference_traces_used=False))
    print(f'Prepared {len(result)} questions: {output / "manifest.json"}')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,default=ROOT/'manifests/training_2000.selection.json')
    p.add_argument('--output',type=Path,default=ROOT/'data/train2000')
    p.add_argument('--cache',type=Path,default=ROOT/'cache/parquet-ranges')
    a=p.parse_args();materialize(a.selection,a.output,a.cache)

if __name__=='__main__':main()
