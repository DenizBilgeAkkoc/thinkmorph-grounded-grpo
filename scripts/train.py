"""Portable single-node training launcher with resumable phases and W&B media."""
import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]

def read(p):
    for attempt in range(4):
        try:return json.loads(Path(p).read_text())
        except (OSError,json.JSONDecodeError) as error:
            if attempt==3 or not (isinstance(error,json.JSONDecodeError) or getattr(error,'errno',None) in (errno.ESTALE,errno.ENOENT,errno.EINTR)):raise
            time.sleep(.05*2**attempt)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def freeze(p,value):
    if p.exists():
        if read(p)!=value:raise ValueError('Frozen run input differs: '+str(p))
    else:write(p,value)

def committed_checkpoint(checkpoint):
    checkpoint=Path(checkpoint)
    for path in (checkpoint,Path(str(checkpoint)+'.rng.pt')):
        if not path.is_file():raise ValueError('Missing checkpoint state: '+str(path))
    marker=read(Path(str(checkpoint)+'.complete.json'))
    proof=read(checkpoint.parent/'checkpoint_validation.json')
    update=read(checkpoint.parent/'update.json')
    if not all(proof.get(k) is True for k in ('reload_passed','optimizer_state_exact','per_rank_rng_restored')):
        raise ValueError('Resume requires completed checkpoint reload proof')
    for key in ('policy_digest','optimizer_digest','world_size'):
        if proof.get(key)!=marker.get(key):raise ValueError('Reload proof does not match checkpoint: '+key)
    for key in ('policy_digest','optimizer_digest','optimizer_updates','wave','sample_history_digest'):
        if update.get(key)!=marker.get(key):raise ValueError('Committed update does not match checkpoint: '+key)
    if update['last_group']!=marker['group']:raise ValueError('Committed checkpoint group mismatch')
    return marker

def prepare(a):
    run=a.run_dir.resolve();run.mkdir(parents=True,exist_ok=True);(run/'config').mkdir(exist_ok=True);(run/'logs').mkdir(exist_ok=True)
    profile=read(ROOT/'configs/profiles.json')[a.profile]
    rows=read(a.manifest)
    if not 0<a.questions<=len(rows):raise ValueError('questions must select a nonempty manifest prefix')
    if a.gpus<1 or a.judge_workers<1:raise ValueError('GPU and judge worker counts must be positive')
    if a.gemini_key_file and not a.gemini_key_file.is_file():raise ValueError('Gemini key file does not exist')
    if not a.prepare_only and a.judge_provider=='gemini' and not (a.gemini_key_file or os.environ.get('GEMINI_API_KEY')):raise ValueError('Supply GEMINI_API_KEY or --gemini-key-file')
    if a.max_updates is not None and a.max_updates<1:raise ValueError('max-updates must be positive')
    if not a.prepare_only:
        for path in [a.native_source/'modeling',a.model_dir/'model.safetensors',a.model_dir/'ae.safetensors']:
            if not path.exists():raise ValueError('Missing required asset: '+str(path))
    for row in rows:
        path=Path(row['problem_image']['path'])
        if not path.is_absolute() or not path.is_file():raise ValueError('Materialize the input images with prepare_data.py first')
        if sha(path)!=row['problem_image']['sha256']:raise ValueError('Input image checksum changed: '+str(path))
    preview=read(ROOT/'configs/preview.template.json')
    preview.update(source=str(a.native_source.resolve()),checkpoint=str(a.model_dir.resolve()))
    recipe=read(ROOT/'configs/training.json')
    if a.judge_provider=='qwen':
        if not a.judge_url:raise ValueError('Qwen requires --judge-url')
        recipe.update(judge_model='Qwen/Qwen3-VL-32B-Instruct',judge_provider='local_qwen')
    else:recipe['judge_model']=a.gemini_model
    frozen=dict(profile=a.profile,profile_settings=profile,world_size=a.gpus,judge_provider=a.judge_provider,
        judge_workers=a.judge_workers,judge_timeout=a.judge_timeout,judge_http2=a.judge_http2,
        cpu_threads=a.cpu_threads,manifest_sha256=sha(a.manifest),
        source_hashes={p.name:sha(p) for p in sorted((ROOT/profile['source']/'thinkmorph_grpo').glob('*.py'))})
    for name,value in [('preview.json',preview),('training.json',recipe),('manifest.json',rows),('run_contract.json',frozen)]:freeze(run/'config'/name,value)
    segments=read(run/'Segments.json') if (run/'Segments.json').exists() else dict(fixed_manifest_sha256=sha(run/'config/manifest.json'),segments=[])
    if segments['segments'] and not a.resume:raise ValueError('Existing run requires explicit --resume; it will not start again from base')
    if a.resume:
        marker=committed_checkpoint(a.resume)
        if marker['world_size']!=a.gpus:raise ValueError('Changing world size requires an audited checkpoint migration')
        for segment in segments['segments']:
            for path in Path(segment['path']).glob('wave_*/checkpoint.pt.complete.json'):
                if read(path)['group']>marker['group']:raise ValueError('Resume must use the latest committed checkpoint; rollback would repeat samples')
        start=marker['group']+1
        if start>=a.questions:raise ValueError('No new questions selected after resume')
    else:start=0
    phase=run/f"phase_{len(segments['segments']):06d}"
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1',f'--nproc-per-node={a.gpus}',
        '--max-restarts=0',str(ROOT/'scripts/worker.py'),'--preview-config',str(run/'config/preview.json'),
        '--recipe',str(run/'config/training.json'),'--manifest',str(run/'config/manifest.json'),
        '--output',str(phase),'--rollout-batch-size',str(profile['text_batch_size']),
        '--max-groups',str(a.questions),'--judge-provider',a.judge_provider,
        '--judge-observation-mode','direct','--judge-workers',str(a.judge_workers),'--judge-timeout',str(a.judge_timeout),
        '--cpu-threads',str(a.cpu_threads),'--gradient-bucket-mb','16','--distributed-timeout-minutes','60',
        '--audit-judge-rpc','--skip-unscorable-groups','--verify-checkpoint-reload']
    if a.judge_provider=='gemini':
        command+=['--gemini-model',a.gemini_model,'--gemini-thinking','low']
        if a.gemini_key_file:command+=['--gemini-api-key-file',str(a.gemini_key_file.resolve())]
    else:command+=['--judge-url',a.judge_url]
    if a.max_updates is not None:command+=['--max-updates',str(a.max_updates)]
    if a.resume:command+=['--resume',str(a.resume.resolve())]
    audit_directories=list(a.resume_audit_directory)
    if a.resume:
        audit_directories += [Path(s['path'])/'sample_history' for s in segments['segments'] if (Path(s['path'])/'sample_history').is_dir()]
    for path in dict.fromkeys(p.resolve() for p in audit_directories):command+=['--resume-audit-directory',str(path)]
    env=dict(os.environ,THINKMORPH_PROFILE=a.profile,THINKMORPH_TEXT_BATCH_SIZE=str(profile['text_batch_size']),
        THINKMORPH_IMAGE_BATCH_SIZE=str(profile['image_batch_size']),THINKMORPH_VAE_DEVICE=profile['vae_device'],
        THINKMORPH_REPLAY_CHECKPOINT=profile['replay_checkpoint'],THINKMORPH_JUDGE_HTTP2=str(int(a.judge_http2)),
        OMP_NUM_THREADS=str(a.cpu_threads),OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',
        PYTHONUNBUFFERED='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    env['PYTHONPATH']=str(ROOT/profile['source'])
    write(run/'Prepared_Command.json',dict(command=command,profile=a.profile,phase=str(phase),start_group=start,
        environment={k:env[k] for k in ['THINKMORPH_PROFILE','THINKMORPH_TEXT_BATCH_SIZE','THINKMORPH_IMAGE_BATCH_SIZE','THINKMORPH_VAE_DEVICE','THINKMORPH_REPLAY_CHECKPOINT','THINKMORPH_JUDGE_HTTP2']}))
    return run,phase,segments,start,command,env

def stop(process):
    if process and process.poll() is None:
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=30)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()

def main():
    def interrupted(signum, frame):raise KeyboardInterrupt('Launcher interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--profile',choices=['production','optimized'],default='optimized')
    p.add_argument('--native-source',type=Path,default=ROOT/'vendor/ThinkMorph')
    p.add_argument('--model-dir',type=Path,default=ROOT/'models/ThinkMorph-7B')
    p.add_argument('--gpus',type=int,default=2);p.add_argument('--questions',type=int,default=4)
    p.add_argument('--max-updates',type=int);p.add_argument('--cpu-threads',type=int,default=6)
    p.add_argument('--judge-provider',choices=['gemini','qwen'],default='gemini');p.add_argument('--judge-url')
    p.add_argument('--gemini-model',default='gemini-3.8-flash');p.add_argument('--gemini-key-file',type=Path)
    p.add_argument('--judge-workers',type=int,default=8);p.add_argument('--judge-timeout',type=float,default=180)
    p.add_argument('--judge-http2',action='store_true');p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--resume',type=Path);p.add_argument('--resume-audit-directory',type=Path,action='append',default=[])
    p.add_argument('--wandb-key-file',type=Path);p.add_argument('--wandb-entity');p.add_argument('--wandb-project',default='thinkmorph-grounded-grpo')
    a=p.parse_args()
    if bool(a.wandb_key_file)!=bool(a.wandb_entity):p.error('Supply both --wandb-key-file and --wandb-entity')
    if a.wandb_key_file and not a.wandb_key_file.is_file():p.error('W&B key file does not exist')
    # Exclude simultaneous controllers for this run, including prepare-only calls.
    import fcntl
    a.run_dir.mkdir(parents=True,exist_ok=True)
    with (a.run_dir/'launcher.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        run,phase,segments,start,command,env=prepare(a)
        if a.prepare_only:print('Prepared configuration and command: '+str(run/'Prepared_Command.json'));return
        segments['segments'].append(dict(id=phase.name,path=str(phase),start_group=start,stop_group=a.questions))
        write(run/'Segments.json',segments)
        controller=dict(status='running',phase=phase.name,active_phase_directory=str(phase),committed_questions=start,
            target_questions=a.questions,
            optimizer_updates=read(Path(str(a.resume)+'.complete.json'))['optimizer_updates'] if a.resume else 0)
        def publish():
            for q in sorted(phase.glob('wave_*/update.json')):
                if (q.parent/'checkpoint_validation.json').exists():
                    w=read(q);marker=committed_checkpoint(w['checkpoint'])
                    controller.update(committed_questions=w['last_group']+1,optimizer_updates=w['optimizer_updates'])
            write(run/'Controller_Status.json',{**controller,'updated_at_unix':time.time()})
        publish();trainer=logger=None;handles=[]
        try:
            log=(run/'logs'/f'{phase.name}.log').open('a');handles.append(log)
            trainer=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            if a.wandb_key_file:
                log=(run/'logs/wandb.log').open('a');handles.append(log)
                logger=subprocess.Popen([sys.executable,'-u',str(ROOT/'logging/wandb_sidecar.py'),'--run-root',str(run),
                    '--manifest',str(run/'config/manifest.json'),'--key-file',str(a.wandb_key_file.resolve()),
                    '--entity',a.wandb_entity,'--project',a.wandb_project,'--name',run.name],
                    env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while trainer.poll() is None:publish();time.sleep(10)
            controller['status']='completed' if trainer.returncode==0 else 'failed';publish()
            if logger:
                try:logger.wait(timeout=60)
                except subprocess.TimeoutExpired:stop(logger)
                write(run/'WandB_Logger_Exit.json',dict(returncode=logger.returncode))
            if trainer.returncode:raise RuntimeError('Training failed; see '+str(run/'logs'/f'{phase.name}.log'))
        except BaseException:
            controller['status']='failed';publish();raise
        finally:
            stop(trainer);stop(logger)
            for log in handles:log.close()

if __name__=='__main__':main()
