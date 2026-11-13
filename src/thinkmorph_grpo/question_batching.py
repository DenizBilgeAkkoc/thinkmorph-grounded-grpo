"""Pack text from several independent N8 questions into one GPU forward.

Image/SDE operations stay singleton. Each question has its own reward group,
seed range and saved traces. Question-local artifacts are explicit projections
of a hash-bound global sampling artifact; they are not claimed to be the actual
sampling packing. Training retains the existing pair-of-two full-causal scorer.
"""
from copy import deepcopy
from pathlib import Path
import json,time
import torch
from PIL import Image
from . import paired_rollout as pr
from .paired_sampling import sample_paired_text

REVISION='cross_question_text_v1'

def project_sample(sampled, rows):
    if not rows or len(set(rows))!=len(rows) or rows!=sorted(rows):
        raise ValueError('Projection needs unique ordered global rows')
    if rows[-1]>=len(sampled['lanes']) or rows[0]<0:raise ValueError('Projection row out of bounds')
    lanes=[]
    for local,global_row in enumerate(rows):
        lane=deepcopy(sampled['lanes'][global_row]);lane['lane_index']=local;lanes.append(lane)
    counts=[x['sampled_tokens'] for x in lanes]
    schedule=[[i for i,c in enumerate(counts) if t<c] for t in range(max(counts))]
    reverse={global_row:local for local,global_row in enumerate(rows)}
    projected=[[reverse[i] for i in active if i in reverse] for active in sampled['active_lane_schedule']]
    if [x for x in projected if x]!=schedule:raise ValueError('Global/local stopping schedule mismatch')
    return dict(layout_revision=sampled['layout_revision'],lanes=lanes,active_lane_schedule=schedule,
                model_forward_count=len(schedule),total_sampled_tokens=sum(counts))

def validate_global(document):
    if document.get('revision')!=REVISION or document.get('kind')!='cross_question_text_batch':
        raise ValueError('Unsupported cross-question sampling artifact')
    members=document['members']
    if not 1<=len(members)<=32:raise ValueError('Invalid global batch size')
    identities=[(m['group'],m['rollout_index']) for m in members]
    if identities!=sorted(set(identities)):raise ValueError('Repeated or reordered cross-question lane')
    counts=[]
    for row,m in enumerate(members):
        if m['row']!=row or type(m['group']) is not int or not 0<=m['rollout_index']<8:
            raise ValueError('Invalid global lane identity')
        if m['seed']!=document['base_seed']+1000*m['group']+m['rollout_index']:
            raise ValueError('Global lane seed crossed question boundary')
        pr.validate_event(m['sampling_values'],document);counts.append(m['sampling_values']['sampled_tokens'])
    schedule=[[i for i,c in enumerate(counts) if t<c] for t in range(max(counts))]
    if (document['active_lane_schedule']!=schedule or document['model_forward_count']!=len(schedule)
            or document['total_sampled_tokens']!=sum(counts)):
        raise ValueError('Global schedule dropped an EOS/cap draw')
    if document['active_questions']!=sorted({x[0] for x in identities}):raise ValueError('Global question set changed')

def validate_projection(group_root,info,artifact):
    reference=artifact.get('cross_question_sampling')
    if not info.get('cross_question_sampling') or not isinstance(reference,dict):
        raise ValueError('Missing explicit cross-question projection identity')
    if reference.get('revision')!=REVISION:raise ValueError('Unsupported projection revision')
    path=pr.safe_child(Path(group_root).parent,reference['path'],within=Path(group_root).parent/'mixed_batches')
    if pr.file_sha(path)!=reference['sha256']:raise ValueError('Global sampling artifact hash mismatch')
    document=json.loads(path.read_text());validate_global(document)
    declaration=info['configuration'].get('question_batch')
    if (not isinstance(declaration,dict) or document['question_batch']!=declaration
            or info['cross_question_sampling']!=declaration):raise ValueError('Cross-question configuration changed')
    for key in ('policy_digest','prior_optimizer_updates','source_fingerprint','rollout_configuration_digest',
                'experimental_configuration_digest'):
        if document[key]!=info[key]:raise ValueError('Global sampling identity differs: '+key)
    for key in ('bos_id','eos_id','max_length','temperature','layout_revision'):
        if document[key]!=artifact[key]:raise ValueError('Global generation setting differs: '+key)
    expected=[m['row'] for m in document['members'] if m['group']==info['group']]
    if reference['global_rows']!=expected or len(expected)!=len(artifact['members']):
        raise ValueError('Projection omitted or imported another question lane')
    for local,global_row in zip(artifact['members'],expected):
        member=document['members'][global_row]
        for key in ('rollout_index','seed','prefix_kv_lens','prefix_ropes'):
            if member[key]!=local[key]:raise ValueError('Global/local lane mismatch: '+key)
        for key in pr.EVENT_KEYS:
            if member['sampling_values'][key]!=local['event_core'][key]:
                raise ValueError('Global/local sampled token or logp mismatch')
    sidecar=pr.safe_child(path.parent,document['rng_sidecar']['file'],within=path.parent)
    if pr.file_sha(sidecar)!=document['rng_sidecar']['sha256']:raise ValueError('Global RNG sidecar changed')
    reverse={row:i for i,row in enumerate(expected)}
    schedule=[[reverse[row] for row in step if row in reverse] for step in document['active_lane_schedule']]
    if [step for step in schedule if step]!=artifact['active_lane_schedule']:
        raise ValueError('Question schedule is not the actual global projection')

@torch.no_grad()
def generate_questions(loaded,*,cfg,requests,output,experiment_configuration,policy_digest,
                       prior_optimizer_updates=0,progress=None,on_rollout_start=None,on_rollout_complete=None):
    from . import train as base
    from .sde import shared_windows
    from data.data_utils import pil_img2rgb
    if len(requests) not in (1,2,4):raise ValueError('Bounded trials support 1, 2, or 4 questions per GPU')
    indexes=[x['group_index'] for x in requests]
    if indexes!=sorted(set(indexes)) or any(type(i) is not int or i<0 for i in indexes):
        raise ValueError('Question identities must be unique ordered nonnegative indices')
    if pr.TEXT_BATCH_SIZE!=8:raise ValueError('Each projected question must retain eight logical rollout lanes')
    settings=cfg['native_settings'];base.require_no_cfg(settings,'Cross-question rollout')
    if cfg['group_size']!=8 or not settings['do_sample'] or settings['understanding_output'] or settings.get('enable_taylorseer',False):
        raise ValueError('Unsupported native sampling settings')
    if experiment_configuration['rollout']!=cfg or pr.source_identity(base,cfg)!=experiment_configuration['source_identity']:
        raise ValueError('Sampling source/configuration changed')
    if base.adapter_digest(loaded[0])!=policy_digest:raise ValueError('Policy changed before question batch')
    declaration=dict(revision=REVISION,group_indexes=indexes,questions=len(requests),max_text_lanes=8*len(requests),
                     reward_group_size=8,replay_grouping='within_question_pairs_of_two',singleton_images=pr.ROLLOUT_EXECUTION['singleton_images'])
    experiment=deepcopy(experiment_configuration);experiment['question_batch']=declaration
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False);(root/'mixed_batches').mkdir()
    model=loaded[0];lanes=[];groups={};started=time.perf_counter()
    native_hooks=(model.generate_image,loaded[1].encode)
    def emit(event,**fields):
        if progress:progress(dict(event=event,**fields))
    try:
        for request in requests:
            group=request['group_index'];example=request['example'];folder=root/f'group_{group:06d}';folder.mkdir()
            image_path=Path(example['problem_image']['path']).resolve();image_hash=pr.file_sha(image_path)
            if example['problem_image'].get('sha256') not in (None,image_hash):raise ValueError('Input image changed')
            with Image.open(image_path) as source:image=source.convert('RGB')
            windows=shared_windows(cfg['seed']+group,settings['max_rounds'],settings['num_timesteps'],cfg['sde_window_size'])
            info=dict(schema_version=pr.SCHEMA_VERSION,experimental_paired_text=True,group=group,
                example_id=example.get('example_id',example['task']),example=example,input_image_sha256=image_hash,
                policy_digest=policy_digest,special_token_ids=deepcopy(loaded[5]),prior_optimizer_updates=prior_optimizer_updates,
                group_size=8,completed_rollouts=0,rollout_configuration_digest=experiment['rollout_configuration_digest'],
                experimental_configuration_digest=pr.canonical_sha(experiment),source_fingerprint=experiment['source_fingerprint'],
                configuration=experiment,status='sampling',started_at_unix=time.time(),pairs=[],windows=windows,
                cross_question_sampling=declaration,planned_rollouts=[dict(rollout_index=n,folder=f'rollout_{n:02d}',
                    seed=cfg['seed']+group*1000+n,status='planned') for n in range(8)])
            groups[group]=dict(info=info,folder=folder,folders=[],text_batches=0,image_path=image_path)
            pr.write_json(folder/'group.json',info)
            for number in range(8):
                path=folder/f'rollout_{number:02d}';path.mkdir();seed=cfg['seed']+group*1000+number
                if on_rollout_start:on_rollout_start(group,number,seed,list(windows))
                lane=pr.initialize_lane(loaded,folder=path,rollout_index=number,seed=seed,windows=windows,cfg=cfg,image=image,question=example['question'])
                lane['group']=group;lane['sampling']=pr.sampling_metadata(lane,cfg,example,group,policy_digest,experiment,prior_optimizer_updates)
                lane['sampling']['cross_question_sampling']=declaration
                pr.write_json(path/'sampling.json',lane['sampling']);lanes.append(lane);groups[group]['folders'].append(path)
                info['planned_rollouts'][number]['status']='sampling'
            pr.write_json(folder/'group.json',info)
        round_index=0
        while any(lane['active'] for lane in lanes):
            active=[lane for lane in lanes if lane['active']];rng_before=[lane['stream'].snapshot() for lane in active]
            generators=[lane['stream'].generator(next(model.parameters()).device) for lane in active]
            emit('mixed_text_started',round=round_index,lanes=len(active),questions=sorted({l['group'] for l in active}))
            def token_progress(state):
                if (state['step']+1)%128==0:emit('mixed_text_progress',round=round_index,step=state['step']+1,active_lanes=len(state['active_lanes']))
            t=time.perf_counter();sampled=sample_paired_text(model,[l['context'] for l in active],bos_id=loaded[5]['bos_token_id'],
                eos_id=loaded[5]['eos_token_id'],max_length=settings['max_think_token_n'],temperature=settings['text_temperature'],
                generators=generators,progress=token_progress);text_seconds=time.perf_counter()-t
            for lane,generator in zip(active,generators):lane['stream'].adopt_generator(generator)
            stem=f'batch_{round_index:03d}';rng_path=root/'mixed_batches'/(stem+'_rng.pt')
            pr.torch_save(rng_path,dict(members=[dict(group=l['group'],rollout_index=l['rollout_index'],before=before,after=l['stream'].snapshot()) for l,before in zip(active,rng_before)]))
            document=dict(kind='cross_question_text_batch',revision=REVISION,question_batch=declaration,base_seed=cfg['seed'],
                active_questions=sorted({l['group'] for l in active}),policy_digest=policy_digest,prior_optimizer_updates=prior_optimizer_updates,
                source_fingerprint=experiment['source_fingerprint'],rollout_configuration_digest=experiment['rollout_configuration_digest'],
                experimental_configuration_digest=pr.canonical_sha(experiment),bos_id=loaded[5]['bos_token_id'],eos_id=loaded[5]['eos_token_id'],
                max_length=settings['max_think_token_n'],temperature=settings['text_temperature'],layout_revision=sampled['layout_revision'],
                active_lane_schedule=sampled['active_lane_schedule'],model_forward_count=sampled['model_forward_count'],
                total_sampled_tokens=sampled['total_sampled_tokens'],seconds=text_seconds,
                rng_sidecar=dict(file=rng_path.name,sha256=pr.file_sha(rng_path)),members=[dict(row=i,group=l['group'],rollout_index=l['rollout_index'],
                    seed=l['seed'],prefix_kv_lens=list(l['context']['kv_lens']),prefix_ropes=list(l['context']['ropes']),
                    sampling_values={k:deepcopy(sampled['lanes'][i][k]) for k in pr.EVENT_KEYS}) for i,l in enumerate(active)])
            validate_global(document);global_path=root/'mixed_batches'/(stem+'.json');pr.write_json(global_path,document)
            cores_by_identity={}
            for group in document['active_questions']:
                rows=[i for i,l in enumerate(active) if l['group']==group];local=[active[i] for i in rows];g=groups[group]
                projection=project_sample(sampled,rows)
                reference=dict(revision=REVISION,path=global_path.relative_to(root).as_posix(),sha256=pr.file_sha(global_path),global_rows=rows,
                               seconds_are_shared_across_questions=True)
                artifact,cores=pr.write_text_batch(g['folder'],0,g['text_batches'],local,projection,[rng_before[i] for i in rows],
                    text_seconds,cfg,g['info'],sampling_context=reference)
                g['text_batches']+=1
                for lane,core in zip(local,cores):cores_by_identity[(group,lane['rollout_index'])]=core
            from .image_batching import advance_lanes
            advance_lanes(active, [cores_by_identity[(l['group'],l['rollout_index'])] for l in active], settings,
                {g['folder']:g['info'] for g in groups.values()}, progress=progress)
            for lane in active:
                g=groups[lane['group']]
                if not lane['active']:
                    lane['sampling'].update(status='complete',completed_at_unix=time.time(),generated_images=lane['rounds'],text_batches=lane['text_batches'],
                        termination_reason=lane['termination_reason'],prefix_seconds=lane['prefix_seconds'])
                    pr.torch_save(lane['folder']/'rng_final.pt',lane['stream'].snapshot())
                    for suffix in ('initial','final'):lane['sampling']['rng_'+suffix+'_sha256']=pr.file_sha(lane['folder']/('rng_'+suffix+'.pt'))
                    pr.write_json(lane['folder']/'sampling.json',lane['sampling']);g['info']['completed_rollouts']+=1
                    g['info']['planned_rollouts'][lane['rollout_index']]['status']='complete';pr.write_json(g['folder']/'group.json',g['info'])
                    if on_rollout_complete:on_rollout_complete(lane['group'],lane['folder'],lane['rollout_index'],lane['seed'])
                    for key in ('context','cfg_text_context','cfg_image_context'):lane[key]=None
            round_index+=1
        torch.cuda.synchronize();seconds=time.perf_counter()-started;lanes[-1]['stream'].restore_process()
        if native_hooks!=(model.generate_image,loaded[1].encode) or base.adapter_digest(model)!=policy_digest or pr.source_identity(base,cfg)!=experiment['source_identity']:
            raise RuntimeError('Native hooks, policy, or source changed during batch')
        results=[]
        for group,g in groups.items():
            if pr.file_sha(g['image_path'])!=g['info']['input_image_sha256']:raise RuntimeError('Input image changed')
            info=g['info'];info.update(status='complete',completed_at_unix=time.time(),sampling_seconds=seconds,sampling_seconds_shared_across_questions=True,
                policy_unchanged=True,optimizer_steps=0,judge_calls_within_generator=0,pairs=[dict(pair_index=0,text_batches=g['text_batches'],rollout_indexes=list(range(8)))])
            pr.write_json(g['folder']/'group.json',info);traces=[json.loads((p/'trace.json').read_text()) for p in g['folders']]
            batches=pr.load_text_batches(g['folders'],traces);results.append(dict(group=group,folders=g['folders'],traces=traces,text_batches=batches))
        pr.write_json(root/'batch_summary.json',dict(status='complete',revision=REVISION,questions=indexes,rollouts=8*len(requests),seconds=seconds,source_fingerprint=experiment['source_fingerprint']))
        return dict(groups=results,seconds=seconds)
    except BaseException as error:
        pr.write_json(root/'batch_failure.json',dict(status='failed',error_type=type(error).__name__,error=str(error)))
        raise
