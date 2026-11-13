"""Experimental same-question diffusion batches with hash-bound matched replay.

Each image retains its own CPU RNG, transitions, reward and loss denominator.
Only native flow forwards are packed. VAE/context work remains singleton.
"""
from copy import deepcopy
import inspect
import json
from pathlib import Path
import time
import torch
from . import paired_rollout as pr
from .paired_packing import pack_contexts
from .sde import schedule, sde_step, ode_step
from .guidance import require_no_cfg

REVISION = 'same_question_image_batch_v1'
IDENTITIES = ('group','example_id','policy_digest','prior_optimizer_updates',
              'source_fingerprint','experimental_configuration_digest','rollout_configuration_digest')


def packed_inputs(model, infer, contexts, events):
    if not contexts or len(contexts) != len(events):
        raise ValueError('Missing image batch contexts')
    settings = events[0]['settings']
    require_no_cfg(settings, 'batched image flow')
    if any(e['settings'] != settings for e in events):
        raise ValueError('Image batch settings differ')
    packed = pack_contexts(contexts)
    # Geometry construction draws unused noise. Preserve the caller's CPU RNG.
    with torch.random.fork_rng(devices=[]):
        inputs = model.prepare_vae_latent(curr_kvlens=packed['kv_lens'], curr_rope=packed['ropes'],
            image_sizes=[e['image_shape'] for e in events], new_token_ids=infer.new_token_ids)
    inputs.pop('packed_init_noises')
    inputs['past_key_values'] = packed['past_key_values']
    for key in ('cfg_text_scale','cfg_img_scale','cfg_renorm_min','cfg_renorm_type'):
        inputs[key] = settings[key]
    return inputs


def velocities(model, inputs, xs, t):
    model.language_model.model.enable_taylorseer = False
    x = torch.cat(xs)
    fn = inspect.unwrap(type(model)._forward_flow)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        v = fn(model, x_t=x, timestep=t.expand(len(x)), **inputs).to(x.device)
    return list(v.split([len(a) for a in xs]))


@torch.no_grad()
def generate_batch(lanes, settings, info):
    model, infer = lanes[0]['bridge'].model, lanes[0]['recorder']
    root = lanes[0]['folder'].parent
    ordinal = lanes[0]['rounds']
    if any(l['folder'].parent != root or l['rounds'] != ordinal for l in lanes):
        raise ValueError('Only the same question and image ordinal may share a flow batch')
    directory = root/'image_batches'; directory.mkdir(exist_ok=True)
    number = len(list(directory.glob('batch_*.json')))
    path = directory/f'batch_{number:06d}.json'
    if path.exists(): raise FileExistsError(path)
    batch_id = f'g{info["group"]:06d}_i{number:06d}'
    before = [l['stream'].snapshot() for l in lanes]
    cores, xs = [], []
    started = time.perf_counter()
    for lane in lanes:
        with lane['stream'].scope():
            noise = model.prepare_vae_latent(curr_kvlens=lane['context']['kv_lens'],
                curr_rope=lane['context']['ropes'], image_sizes=[lane['image_shape']],
                new_token_ids=infer.new_token_ids)['packed_init_noises']
        xs.append(noise)
        cores.append(dict(kind='image',image_shape=list(lane['image_shape']),
            settings={key: settings[key] for key in pr.IMAGE_SETTINGS},
            contexts={key:deepcopy(lane[field]['events']) for key,field in
                (('conditional','context'),('cfg_text','cfg_text_context'),('cfg_image','cfg_image_context'))}))
    inputs = packed_inputs(model,infer,[l['context'] for l in lanes],cores)
    generators = [l['stream'].generator('cpu') for l in lanes]
    ts = schedule(settings['num_timesteps'],settings['timestep_shift'],xs[0].device)
    states = []
    for row,(lane,x) in enumerate(zip(lanes,xs)):
        bridge = lane['bridge']
        start,width = bridge.windows[ordinal],bridge.window_size
        if not 0 <= start < start+width <= len(ts)-1: raise ValueError('Invalid SDE window')
        states.append(dict(window_start=start,window_size=width,
            guidance={k:settings[k] for k in ('cfg_text_scale','cfg_img_scale')},
            schedule=ts.cpu(),noise_level=bridge.noise_level,initial_noise=x.clone(),transitions=[],
            image_batch=dict(revision=REVISION,batch_id=batch_id,row=row)))
    if any((s['window_start'],s['window_size']) != (states[0]['window_start'],states[0]['window_size']) for s in states):
        raise ValueError('Members do not share the same stochastic window')
    flow_started = time.perf_counter()
    for step,t in enumerate(ts[:-1]):
        vs = velocities(model,inputs,xs,t)
        for row,(s,v) in enumerate(zip(states,vs)):
            if s['window_start'] <= step < s['window_start']+s['window_size']:
                prev = xs[row].clone()
                xs[row],lp,_,_ = sde_step(v,t,ts[step+1],xs[row],sigma_max=ts[1],
                    noise_level=s['noise_level'],generator=generators[row])
                s['transitions'].append(dict(index=step,x=prev,next_x=xs[row].clone(),
                    old_logp=lp.detach().cpu(),velocity=v.detach().float().cpu()))
            else: xs[row] = ode_step(v,t,ts[step+1],xs[row])
    flow_seconds = time.perf_counter()-flow_started
    del inputs,vs
    members = []
    for row,(lane,state,core,x,generator) in enumerate(zip(lanes,states,cores,xs,generators)):
        lane['stream'].adopt_generator(generator)
        recorder = lane['recorder']
        with lane['stream'].scope(), torch.autocast('cuda',dtype=torch.bfloat16):
            generated = recorder.decode_image(x,lane['image_shape'])
            file = recorder.save_image(generated)
        state_file = f'flow_{ordinal:02d}.pt'
        pr.torch_save(lane['folder']/state_file,state)
        lane['bridge'].records.append(state)
        core.update(file=file,flow_file=state_file,seconds=(time.perf_counter()-started)/len(lanes))
        members.append(dict(row=row,rollout_index=lane['rollout_index'],folder=lane['folder'].name,
            trace_event_index=len(recorder.events),image_ordinal=ordinal,event_core=core,
            event_core_sha256=pr.canonical_sha(core),flow_sha256=pr.file_sha(lane['folder']/state_file),
            prefix_assets=pr.prefix_assets(lane['folder'],core['contexts']['conditional']),
            image_sha256=pr.file_sha(lane['folder']/file),
            prefix_kv_lens=lane['context']['kv_lens'],prefix_ropes=lane['context']['ropes']))
        lane['_generated_image'] = generated
    sidecar = path.with_suffix('.pt')
    pr.torch_save(sidecar,dict(batch_id=batch_id,members=[dict(row=i,before=b,after=l['stream'].snapshot())
        for i,(b,l) in enumerate(zip(before,lanes))]))
    artifact = dict(revision=REVISION,batch_id=batch_id,batch_index=number,image_ordinal=ordinal,
        members=members,member_count=len(lanes),flow_seconds=flow_seconds,
        seconds=time.perf_counter()-started,
        rng_sidecar=dict(file=sidecar.name,sha256=pr.file_sha(sidecar)),
        **{key:info[key] for key in IDENTITIES})
    pr.write_json(path,artifact)
    for row,(lane,core) in enumerate(zip(lanes,cores)):
        core['batched_image_replay'] = dict(batch_id=batch_id,row=row,
            artifact_path='../image_batches/'+path.name,artifact_sha256=pr.file_sha(path))
        lane['recorder'].events.append(core)
        pr.write_json(lane['folder']/'trace.json',lane['recorder'].events)
    return artifact


@torch.no_grad()
def advance_lanes(lanes, cores, settings, infos, progress=None):
    from data.data_utils import pil_img2rgb
    ready = {}
    for lane,core in zip(lanes,cores):
        with lane['stream'].scope(), torch.autocast('cuda',dtype=torch.bfloat16):
            lane['context'] = lane['recorder'].update_context_text(core['text'],lane['context'])
        if '<image_start>' in core['text']:
            ready.setdefault((lane['folder'].parent,lane['rounds']),[]).append(lane)
        else:
            lane['active'],lane['termination_reason'] = False,'text_without_image_start'
    for (root,ordinal),group in ready.items():
        for start in range(0,len(group),pr.IMAGE_BATCH_SIZE):
            batch = group[start:start+pr.IMAGE_BATCH_SIZE]
            if progress: progress(dict(event='image_batch_started',group=infos[root]['group'],
                image_index=ordinal,rollout_indexes=[l['rollout_index'] for l in batch]))
            artifact = generate_batch(batch,settings,infos[root])
            for lane in batch:
                with lane['stream'].scope(), torch.autocast('cuda',dtype=torch.bfloat16):
                    image = lane.pop('_generated_image')
                    resized = lane['recorder'].vae_transform.resize_transform(pil_img2rgb(image))
                    lane['context'] = lane['recorder'].update_context_image(resized,lane['context'],vae=True)
                lane['rounds'] += 1
                if lane['rounds'] >= settings['max_rounds']:
                    lane['active'],lane['termination_reason'] = False,'max_rounds'
            if progress: progress(dict(event='image_batch_complete',batch_id=artifact['batch_id'],
                images=len(batch),flow_seconds=artifact['flow_seconds']))


def load_image_batches(folders, traces):
    """Validate identities, assets, full coverage and the recorded flow layout."""
    # This also verifies all eight sampling identities and RNG final snapshots.
    pr.load_text_batches(folders,traces)
    folders = [Path(f).resolve() for f in folders]
    root = folders[0].parent
    info = json.loads((root/'group.json').read_text())
    expected,refs = set(),{}
    for n,(folder,trace) in enumerate(zip(folders,traces)):
        for j,event in enumerate(trace):
            if event['kind'] != 'image': continue
            expected.add((n,j))
            ref = event.get('batched_image_replay')
            if not isinstance(ref,dict): raise ValueError('Image lacks matched batch binding')
            path = pr.safe_child(folder,ref['artifact_path'],within=root)
            previous = refs.setdefault(path,ref['artifact_sha256'])
            if previous != ref['artifact_sha256']: raise ValueError('Image batch hash disagreement')
    covered,result = set(),[]
    for path,digest in sorted(refs.items()):
        if pr.file_sha(path) != digest: raise ValueError('Image batch hash mismatch')
        doc = json.loads(path.read_text())
        if doc['revision'] != REVISION or any(doc[k] != info[k] for k in IDENTITIES):
            raise ValueError('Image batch identity mismatch')
        if (path != root/'image_batches'/f'batch_{doc["batch_index"]:06d}.json' or
                doc['batch_id'] != f'g{info["group"]:06d}_i{doc["batch_index"]:06d}'):
            raise ValueError('Image batch path/identity mismatch')
        count = len(doc['members'])
        width = info['configuration']['execution_tuning']['image_batch_size']
        if not 1 <= count <= width or doc['member_count'] != count: raise ValueError('Invalid image batch width')
        side = pr.safe_child(path.parent,doc['rng_sidecar']['file'],within=root)
        if pr.file_sha(side) != doc['rng_sidecar']['sha256']: raise ValueError('Image RNG sidecar changed')
        members,seen = [],[]
        for row,m in enumerate(doc['members']):
            n,j = m['rollout_index'],m['trace_event_index']
            if type(n) is not int or n not in range(8) or n in seen or m['row'] != row:
                raise ValueError('Invalid image member order')
            seen.append(n)
            if type(j) is not int or not 0 <= j < len(traces[n]): raise ValueError('Invalid image event index')
            folder,event = folders[n],traces[n][j]
            core = {k:v for k,v in event.items() if k != 'batched_image_replay'}
            ref = event.get('batched_image_replay',{})
            if (pr.safe_child(root,m['folder']) != folder or event['kind'] != 'image' or
                core != m['event_core'] or pr.canonical_sha(core) != m['event_core_sha256'] or
                ref.get('batch_id') != doc['batch_id'] or ref.get('row') != row or
                ref.get('artifact_sha256') != digest or
                pr.safe_child(folder,ref['artifact_path'],within=root) != path):
                raise ValueError('Image event is not bound to its recorded batch')
            ordinal = sum(e['kind']=='image' for e in traces[n][:j])
            if ordinal != m['image_ordinal'] or ordinal != doc['image_ordinal']:
                raise ValueError('Image ordinal mismatch')
            if (pr.prefix_assets(folder,core['contexts']['conditional']) != m['prefix_assets'] or
                pr.file_sha(pr.safe_child(folder,core['file'])) != m['image_sha256'] or
                pr.file_sha(pr.safe_child(folder,core['flow_file'])) != m['flow_sha256']):
                raise ValueError('Image prefix, output or transition file changed')
            state = torch.load(folder/core['flow_file'],weights_only=True)
            cfg = info['configuration']['rollout']; settings = cfg['native_settings']
            require_no_cfg(core['settings'],'image replay binding')
            if core['settings'] != {k:settings[k] for k in pr.IMAGE_SETTINGS}:
                raise ValueError('Image generation settings mismatch')
            ts = schedule(settings['num_timesteps'],settings['timestep_shift'],'cpu')
            begin = info['windows'][ordinal]; size = cfg['sde_window_size']
            if (state['image_batch'] != dict(revision=REVISION,batch_id=doc['batch_id'],row=row) or
                not torch.equal(state['schedule'],ts) or state['window_start'] != begin or
                state['window_size'] != size or state['noise_level'] != cfg['noise_level'] or
                [t['index'] for t in state['transitions']] != list(range(begin,begin+size))):
                raise ValueError('Image transition layout mismatch')
            if (n,j) in covered: raise ValueError('Duplicate image action')
            covered.add((n,j))
            step_index = sum(e['kind']=='text' for e in traces[n][:j])-1
            members.append(dict(**{k:m[k] for k in ('rollout_index','prefix_kv_lens','prefix_ropes')},
                step_index=step_index,event=event,folder=folder,state=state))
        if seen != sorted(seen): raise ValueError('Image lane order changed')
        result.append(dict(batch_id=doc['batch_id'],members=members))
    if covered != expected: raise ValueError('Incomplete image batch coverage')
    return result


def rebuild_inputs(model, infer, members):
    from . import trainer as native
    contexts = [native.rebuild_context(infer,m['event']['contexts']['conditional'],m['folder']) for m in members]
    for c,m in zip(contexts,members):
        if c['kv_lens'] != m['prefix_kv_lens'] or c['ropes'] != m['prefix_ropes']:
            raise ValueError('Image prefix layout changed during replay')
    return packed_inputs(model,infer,contexts,[m['event'] for m in members])


def transition_logprobs(model, inputs, members, k):
    states = [m['state'] for m in members]
    transitions = [s['transitions'][k] for s in states]
    index = transitions[0]['index']; ts = states[0]['schedule']
    if any(t['index'] != index for t in transitions): raise ValueError('Image transition timestep mismatch')
    vs = velocities(model,inputs,[t['x'] for t in transitions],ts[index])
    logps = [sde_step(v,ts[index],ts[index+1],t['x'],sigma_max=ts[1],
        noise_level=s['noise_level'],previous_sample=t['next_x'])[1]
        for v,s,t in zip(vs,states,transitions)]
    return logps,vs


def backward_images(trainer, batches, advantages, image_count, metrics, details, deltas, optimize):
    from . import trainer as native
    for batch in batches:
        started = time.monotonic(); members = batch['members']
        inputs = rebuild_inputs(trainer.model,trainer.infer,members)
        reference_inputs = None
        if trainer.config.velocity_beta:
            with torch.no_grad(),native.reference_adapters(trainer.model):
                reference_inputs = rebuild_inputs(trainer.model,trainer.infer,members)
        for k in range(len(members[0]['state']['transitions'])):
            logps,vs = transition_logprobs(trainer.model,inputs,members,k)
            refs = None
            if reference_inputs is not None:
                with torch.no_grad(),native.reference_adapters(trainer.model):
                    _,refs = transition_logprobs(trainer.model,reference_inputs,members,k)
            losses = []
            for row,(m,lp,v) in enumerate(zip(members,logps,vs)):
                n,s = m['rollout_index'],m['step_index']; state = m['state']; t = state['transitions'][k]
                saved = native.saved_logprob(t['old_logp'],required=optimize,kind='image')
                old,delta,denom = native.behavior_logprob(lp,saved,required=optimize,kind='batched image')
                if delta is not None: deltas.append(delta)
                policy_loss,ratio = native.ppo_loss(lp,old.to(lp.device),advantages[n,s],trainer.config)
                penalty = native.velocity_regularizer(v,refs[row]) if refs is not None else v.new_zeros(())
                losses.append((policy_loss+trainer.config.velocity_beta*penalty)/(image_count*len(state['transitions'])))
                metrics['image_losses'].append(float(policy_loss.detach()));metrics['ratios'].append(float(ratio))
                metrics['velocity_mse'].append(float(penalty.detach()))
                details.append(dict(rollout=n,step=s,kind='image',transition=t['index'],delta=delta,
                    per_latent_delta=delta/t['x'].numel() if delta is not None else None,
                    denominator_source=denom,batch_id=batch['batch_id'],batch_size=len(members)))
            torch.stack(losses).sum().backward()
            del losses,logps,vs,refs,lp,v,policy_loss,penalty
        elapsed = time.monotonic()-started
        metrics['replay_timing_seconds']['image_replay_and_backward'] += elapsed
        trainer._progress(kind='image',batch_id=batch['batch_id'],batch_size=len(members),
            action_seconds=elapsed,completed_image_transitions=len(metrics['image_losses']),
            sampling_replay_delta=max(deltas,default=0.))
        del inputs,reference_inputs
