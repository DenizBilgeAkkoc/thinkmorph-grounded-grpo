"""Experimental N8 interleaving: paired text, singleton context and image work.

This module does not judge, optimize, modify production, or claim deployment
readiness. It records the actual packed behavior policy and hash-bound replay
layout. Four fixed pairs still form ONE original eight-rollout question group.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import traceback

import torch
from PIL import Image

from .paired_sampling import LaneRandomStream, LAYOUT_REVISION, sample_paired_text
from .replay import TEXT_LIKELIHOOD_BACKEND


SCHEMA_VERSION = 1
TEXT_BATCH_SIZE = int(__import__('os').environ.get('THINKMORPH_TEXT_BATCH_SIZE', '2'))
if TEXT_BATCH_SIZE not in (2, 4, 8):
    raise ValueError('RTX candidate supports text batches 2, 4, 8')
BATCH_MEMBERS = [list(range(i, i + TEXT_BATCH_SIZE)) for i in range(0, 8, TEXT_BATCH_SIZE)]
ROLLOUT_EXECUTION = dict(text_batch_size=TEXT_BATCH_SIZE, layout_revision=LAYOUT_REVISION,
                         singleton_prefixes=True, singleton_images=True)
IMAGE_BATCH_SIZE = int(__import__('os').environ.get('THINKMORPH_IMAGE_BATCH_SIZE','1'))
if IMAGE_BATCH_SIZE not in (1,2,4,8):
    raise ValueError('Image batch size must be 1, 2, 4, or 8')
if IMAGE_BATCH_SIZE > 1:
    ROLLOUT_EXECUTION.update(singleton_images=False,image_batch_size=IMAGE_BATCH_SIZE)
IMAGE_SETTINGS = ('cfg_text_scale', 'cfg_img_scale', 'cfg_interval', 'timestep_shift',
                  'num_timesteps', 'cfg_renorm_min', 'cfg_renorm_type')
EVENT_KEYS = ('input_ids', 'target_ids', 'temperature', 'stop', 'sampled_tokens',
              'sampling_token_logps', 'sampling_logp', 'draw_ids', 'draw_logps',
              'decoded_token_ids')


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                   ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def torch_save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def safe_child(root, relative, *, within=None):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError('Artifact paths must be relative strings')
    root = Path(root).resolve()
    limit = Path(within).resolve() if within is not None else root
    candidate = (root/relative).resolve()
    if candidate == limit or limit not in candidate.parents:
        raise ValueError('Artifact path escapes its authorized data directory')
    return candidate


def event_core(event):
    return {key: value for key, value in event.items() if key != 'paired_replay'}


def prefix_assets(folder, prefix):
    names = sorted({event[key] for event in prefix
                    for key in ('file', 'vae_latent_file') if key in event})
    return [dict(file=name, sha256=file_sha(safe_child(folder, name))) for name in names]


def layout_document(artifact):
    return dict(layout_revision=artifact['layout_revision'], bos_id=artifact['bos_id'],
        eos_id=artifact['eos_id'], max_length=artifact['max_length'],
        temperature=artifact['temperature'], active_lane_schedule=artifact['active_lane_schedule'],
        members=[{key: member[key] for key in ('row', 'rollout_index', 'trace_event_index',
            'text_action_index', 'prefix_kv_lens', 'prefix_ropes', 'event_core_sha256')}
                 for member in artifact['members']])


def source_identity(base, cfg):
    """Hash the actual portable package and native files using relative names."""
    package = Path(__file__).resolve().parent
    native = Path(cfg['source'])
    return dict(package_source={path.name: file_sha(path)
        for path in sorted(package.glob('*.py'))},
        native_source={name: file_sha(native/name) for name in (
            'inferencer.py', 'modeling/bagel/bagel.py',
            'modeling/bagel/qwen2_navit.py', 'modeling/autoencoder.py')})

def make_experiment_configuration(base, cfg, recipe, baseline_configuration_digest):
    identity = source_identity(base, cfg)
    return dict(schema_version=SCHEMA_VERSION, experimental=True,
        text_likelihood_backend=TEXT_LIKELIHOOD_BACKEND, text_rollout_batch_size=TEXT_BATCH_SIZE,
        rollout_execution=deepcopy(ROLLOUT_EXECUTION),
        group_size=8, pairs=BATCH_MEMBERS,
        singleton_prefixes=True, singleton_images=ROLLOUT_EXECUTION['singleton_images'], rollout=deepcopy(cfg),
        lora=deepcopy(recipe['lora']), baseline_configuration_digest=baseline_configuration_digest,
        rollout_configuration_digest=base._state_digest(cfg), source_identity=identity,
        source_fingerprint=canonical_sha(identity),
        optimizer_or_judge_enabled=False, execution_tuning=__import__(__package__+".execution_tuning", fromlist=["execution_settings"]).execution_settings())


def sampling_metadata(lane, cfg, example, group, policy_digest, experiment, prior_updates):
    return dict(seed=lane['seed'], windows=lane['windows'], group=group,
        rollout_index=lane['rollout_index'], base_seed=cfg['seed'],
        policy_digest=policy_digest, prior_optimizer_updates=prior_updates,
        base_checkpoint=cfg['checkpoint'], native_revision=cfg['native_revision'],
        rollout_configuration_digest=experiment['rollout_configuration_digest'],
        experimental_configuration_digest=canonical_sha(experiment),
        source_fingerprint=experiment['source_fingerprint'],
        example_id=example.get('example_id', example['task']),
        layout_revision=LAYOUT_REVISION, experimental_paired_text=True,
        text_rollout_batch_size=TEXT_BATCH_SIZE, group_size=8, status='sampling')


def initialize_lane(loaded, *, folder, rollout_index, seed, windows, cfg, image, question):
    """Mirror native [image, question] prefix construction in an isolated RNG scope."""
    from .bridge import NativeFlowBridge, recorded_inferencer
    from inferencer import GEN_THINK_SYSTEM_PROMPT
    from data.data_utils import pil_img2rgb
    model = loaded[0]
    recorder = recorded_inferencer(loaded[-1])(*loaded[:-1])
    bridge = NativeFlowBridge(model, windows, cfg['noise_level'], cfg['sde_window_size'])
    recorder.begin_recording(folder, bridge)
    stream = LaneRandomStream.from_seed(seed)
    lane = dict(rollout_index=rollout_index, seed=seed, windows=windows,
        recorder=recorder, bridge=bridge, stream=stream, folder=Path(folder),
        rounds=0, active=True, started_at_unix=time.time(), text_batches=0)
    torch_save(Path(folder)/'rng_initial.pt', stream.snapshot())
    started = time.perf_counter()
    with stream.scope(), torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        context = recorder.init_gen_context()
        cfg_text, cfg_image = deepcopy(context), deepcopy(context)
        if cfg['native_settings']['think']:
            context = recorder.update_context_text(GEN_THINK_SYSTEM_PROMPT, context)
            cfg_image = recorder.update_context_text(GEN_THINK_SYSTEM_PROMPT, cfg_image)
        resized = recorder.vae_transform.resize_transform(pil_img2rgb(image))
        context = recorder.update_context_image(resized, context, vae=True)
        shape = resized.size[::-1]
        # The native image and immediately following string branches copy this
        # same context twice. One snapshot preserves both branches' final value.
        cfg_text = deepcopy(context)
        context = recorder.update_context_text(question, context)
        cfg_image = recorder.update_context_text(question, cfg_image)
    lane.update(context=context, cfg_text_context=cfg_text, cfg_image_context=cfg_image,
                image_shape=shape, prefix_seconds=time.perf_counter()-started)
    return lane


def write_text_batch(group_root, pair_index, batch_index, lanes, sampled, rng_before,
                     seconds, cfg, group_info, sampling_context=None):
    """Write immutable artifact first, then reference its exact hash from events."""
    group_root = Path(group_root)
    pair_folder = group_root/f'pair_{pair_index:02d}'
    pair_folder.mkdir(exist_ok=True)
    stem = f'batch_{batch_index:03d}'
    path = pair_folder/(stem+'.json')
    if path.exists():
        raise FileExistsError('A text batch artifact is immutable once written')
    batch_id = f'g{group_info["group"]:06d}_p{pair_index:02d}_b{batch_index:03d}'
    states = dict(schema_version=SCHEMA_VERSION, batch_id=batch_id,
        members=[dict(row=row, rollout_index=lane['rollout_index'], before=rng_before[row],
                      after=lane['stream'].snapshot()) for row, lane in enumerate(lanes)])
    state_path = pair_folder/(stem+'_rng.pt')
    torch_save(state_path, states)
    members, cores = [], []
    for row, (lane, sampled_lane) in enumerate(zip(lanes, sampled['lanes'])):
        recorder = lane['recorder']
        if sampled_lane['lane_index'] != row:
            raise ValueError('Sampler reordered its declared member rows')
        decoded = recorder.tokenizer.decode(sampled_lane['decoded_token_ids'])
        # Exactly the released native gen_text string postprocessing.
        text = decoded.split('<|im_end|>')[0].split('<|im_start|>')[1]
        core = dict(kind='text', text=text, prefix=deepcopy(lane['context']['events']),
                    seconds=seconds, paired_wall_seconds_shared=len(lanes) > 1)
        core.update({key: deepcopy(sampled_lane[key]) for key in EVENT_KEYS})
        cores.append(core)
        members.append(dict(row=row, rollout_index=lane['rollout_index'],
            folder=lane['folder'].relative_to(group_root).as_posix(), seed=lane['seed'],
            trace_event_index=len(recorder.events),
            text_action_index=sum(e['kind'] == 'text' for e in recorder.events),
            prefix_kv_lens=list(lane['context']['kv_lens']),
            prefix_ropes=list(lane['context']['ropes']),
            prefix_assets=prefix_assets(lane['folder'], core['prefix']),
            event_core=core, event_core_sha256=canonical_sha(core)))
    first = lanes[0]['recorder']
    artifact = dict(schema_version=SCHEMA_VERSION, kind='paired_text_batch', batch_id=batch_id,
        group=group_info['group'], example_id=group_info['example_id'],
        pair_index=pair_index, batch_index=batch_index, member_count=len(lanes),
        policy_digest=group_info['policy_digest'], prior_optimizer_updates=group_info['prior_optimizer_updates'],
        rollout_configuration_digest=group_info['rollout_configuration_digest'],
        experimental_configuration_digest=group_info['experimental_configuration_digest'],
        source_fingerprint=group_info['source_fingerprint'], layout_revision=sampled['layout_revision'],
        bos_id=first.new_token_ids['bos_token_id'], eos_id=first.new_token_ids['eos_token_id'],
        max_length=cfg['native_settings']['max_think_token_n'],
        temperature=cfg['native_settings']['text_temperature'],
        active_lane_schedule=sampled['active_lane_schedule'], members=members,
        model_forward_count=sampled['model_forward_count'],
        total_sampled_tokens=sampled['total_sampled_tokens'],
        rng_sidecar=dict(file=state_path.name, sha256=file_sha(state_path)),
        seconds=seconds, path_base='group_root', group_root_from_artifact='..')
    if sampling_context is not None:
        artifact['cross_question_sampling'] = deepcopy(sampling_context)
    artifact['layout_sha256'] = canonical_sha(layout_document(artifact))
    write_json(path, artifact)
    digest = file_sha(path)
    for row, (lane, core) in enumerate(zip(lanes, cores)):
        event = dict(core, paired_replay=dict(batch_id=batch_id, row=row,
            artifact_path=f'../{pair_folder.name}/{path.name}', artifact_sha256=digest))
        lane['recorder'].events.append(event)
        lane['text_batches'] += 1
        write_json(lane['folder']/'trace.json', lane['recorder'].events)
    return artifact, cores


@torch.no_grad()
def generate_group(loaded, *, cfg, example, group_index, output, experiment_configuration,
                   policy_digest, prior_optimizer_updates=0, progress=None,
                   on_rollout_complete=None, on_rollout_start=None):
    """Generate exactly one N8 question group; no reward/optimizer work occurs."""
    from . import train as base
    from .sde import shared_windows
    from data.data_utils import pil_img2rgb
    settings = cfg['native_settings']
    base.require_no_cfg(settings, 'Experimental paired rollout')
    if (cfg['group_size'] != 8 or not settings['do_sample'] or settings['understanding_output'] or
            settings.get('enable_taylorseer', False) or settings['max_rounds'] < 1):
        raise ValueError('Only native N8 stochastic interleaved no-CFG rollout is supported')
    if (experiment_configuration['rollout'] != cfg or
            experiment_configuration.get('rollout_execution') != ROLLOUT_EXECUTION or
            experiment_configuration.get('text_likelihood_backend') != TEXT_LIKELIHOOD_BACKEND):
        raise ValueError('Experiment configuration differs from supplied rollout settings')
    if base.adapter_digest(loaded[0]) != policy_digest:
        raise ValueError('Requested behavior policy does not match the actual adapters')
    if source_identity(base, cfg) != experiment_configuration['source_identity']:
        raise ValueError('Experiment source changed after configuration was constructed')
    group_root = Path(output).resolve()
    group_root.mkdir(parents=True, exist_ok=False)
    image_path = Path(example['problem_image']['path']).resolve()
    image_hash = file_sha(image_path)
    if example['problem_image'].get('sha256') not in (None, image_hash):
        raise ValueError('Question image hash does not match declared provenance')
    with Image.open(image_path) as original_image:
        image = original_image.convert('RGB')
    windows = shared_windows(cfg['seed']+group_index, settings['max_rounds'],
                             settings['num_timesteps'], cfg['sde_window_size'])
    info = dict(schema_version=SCHEMA_VERSION, experimental_paired_text=True,
        group=group_index, example_id=example.get('example_id', example['task']),
        example=example, input_image_sha256=image_hash, policy_digest=policy_digest,
        special_token_ids=deepcopy(loaded[5]),
        prior_optimizer_updates=prior_optimizer_updates, group_size=8, completed_rollouts=0,
        rollout_configuration_digest=experiment_configuration['rollout_configuration_digest'],
        experimental_configuration_digest=canonical_sha(experiment_configuration),
        source_fingerprint=experiment_configuration['source_fingerprint'],
        configuration=experiment_configuration, status='sampling', started_at_unix=time.time(), pairs=[],
        windows=windows, planned_rollouts=[dict(rollout_index=n, folder=f'rollout_{n:02d}',
            seed=cfg['seed']+group_index*1000+n, status='planned') for n in range(8)])
    write_json(group_root/'group.json', info)
    model = loaded[0]
    native_generate_image, native_encode = model.generate_image, loaded[1].encode
    folders = []
    started = time.perf_counter()
    try:
        for pair_index in range(len(BATCH_MEMBERS)):
            lanes = []
            pair_started = time.perf_counter()
            for rollout_index in BATCH_MEMBERS[pair_index]:
                seed = cfg['seed']+group_index*1000+rollout_index
                folder = group_root/f'rollout_{rollout_index:02d}'
                folder.mkdir(exist_ok=False)
                stamp = sampling_metadata(dict(seed=seed, windows=windows, rollout_index=rollout_index),
                    cfg, example, group_index, policy_digest, experiment_configuration, prior_optimizer_updates)
                stamp['status'] = 'initializing_prefix'
                write_json(folder/'sampling.json', stamp)
                info['planned_rollouts'][rollout_index]['status'] = 'initializing_prefix'
                write_json(group_root/'group.json', info)
                if on_rollout_start is not None:
                    on_rollout_start(rollout_index, seed, list(windows))
                lane = initialize_lane(loaded, folder=folder, rollout_index=rollout_index,
                    seed=seed, windows=windows, cfg=cfg, image=image, question=example['question'])
                lane['sampling'] = sampling_metadata(lane, cfg, example, group_index,
                    policy_digest, experiment_configuration, prior_optimizer_updates)
                write_json(folder/'sampling.json', lane['sampling'])
                info['planned_rollouts'][rollout_index]['status'] = 'sampling'
                lanes.append(lane)
                folders.append(folder)
            batch_index = 0
            while any(lane['active'] for lane in lanes):
                active = [lane for lane in lanes if lane['active']]
                rng_before = [lane['stream'].snapshot() for lane in active]
                generators = [lane['stream'].generator(next(model.parameters()).device) for lane in active]
                if progress:
                    progress(dict(event='text_batch_started', group=group_index, pair=pair_index,
                        batch=batch_index, rollout_indexes=[lane['rollout_index'] for lane in active]))
                text_started = time.perf_counter()
                def token_progress(state):
                    if progress and (state['step']+1) % 128 == 0:
                        progress(dict(event='text_batch_progress', pair=pair_index,
                            batch=batch_index, step=state['step']+1,
                            active_rows=state['active_lanes']))
                sampled = sample_paired_text(model, [lane['context'] for lane in active],
                    bos_id=active[0]['recorder'].new_token_ids['bos_token_id'],
                    eos_id=active[0]['recorder'].new_token_ids['eos_token_id'],
                    max_length=settings['max_think_token_n'], temperature=settings['text_temperature'],
                    generators=generators, progress=token_progress)
                text_seconds = time.perf_counter()-text_started
                for lane, generator in zip(active, generators):
                    lane['stream'].adopt_generator(generator)
                artifact, cores = write_text_batch(group_root, pair_index, batch_index, active,
                    sampled, rng_before, text_seconds, cfg, info)
                from .image_batching import advance_lanes
                advance_lanes(active, cores, settings, {group_root: info}, progress=progress)
                for lane, core in zip(active, cores):
                    if not lane['active']:
                        lane['sampling'].update(status='complete', completed_at_unix=time.time(),
                            generated_images=lane['rounds'], text_batches=lane['text_batches'],
                            termination_reason=lane['termination_reason'],
                            prefix_seconds=lane['prefix_seconds'])
                        torch_save(lane['folder']/'rng_final.pt', lane['stream'].snapshot())
                        lane['sampling']['rng_initial_sha256'] = file_sha(lane['folder']/'rng_initial.pt')
                        lane['sampling']['rng_final_sha256'] = file_sha(lane['folder']/'rng_final.pt')
                        write_json(lane['folder']/'sampling.json', lane['sampling'])
                        info['completed_rollouts'] += 1
                        info['planned_rollouts'][lane['rollout_index']]['status'] = 'complete'
                        write_json(group_root/'group.json', info)
                        if on_rollout_complete is not None:
                            on_rollout_complete(lane['folder'], lane['rollout_index'], lane['seed'])
                        # Completed trajectory records remain; release GPU prefixes.
                        for key in ('context', 'cfg_text_context', 'cfg_image_context'):
                            lane[key] = None
                batch_index += 1
                write_json(group_root/'group.json', info)
            lanes[-1]['stream'].restore_process()
            info['pairs'].append(dict(pair_index=pair_index, text_batches=batch_index,
                rollout_indexes=[lane['rollout_index'] for lane in lanes],
                seconds=time.perf_counter()-pair_started))
            write_json(group_root/'group.json', info)
            if progress:
                progress(dict(event='pair_complete', **info['pairs'][-1]))
            del lanes
        torch.cuda.synchronize()
        if (model.generate_image != native_generate_image or loaded[1].encode != native_encode or
                base.adapter_digest(model) != policy_digest or file_sha(image_path) != image_hash or
                source_identity(base, cfg) != experiment_configuration['source_identity']):
            raise RuntimeError('Native hooks, policy, source, or input changed unexpectedly')
        info.update(status='complete', sampling_seconds=time.perf_counter()-started,
            completed_at_unix=time.time(), policy_unchanged=True, optimizer_steps=0,
            judge_calls_within_generator=0,
            externally_managed_rollout_completion_callback=on_rollout_complete is not None)
        write_json(group_root/'group.json', info)
        traces = [json.loads((folder/'trace.json').read_text()) for folder in folders]
        validated = load_text_batches(folders, traces)
        return dict(group_root=group_root, folders=folders, traces=traces,
                    text_batches=validated, summary=info)
    except BaseException as error:
        info.update(status='failed', error_type=type(error).__name__, error=str(error),
                    traceback=traceback.format_exc(), finished_at_unix=time.time())
        write_json(group_root/'group.json', info)
        raise


def validate_event(core, artifact):
    bos, eos, maximum = artifact['bos_id'], artifact['eos_id'], artifact['max_length']
    inputs, targets = core['input_ids'], core['target_ids']
    draws, logps = core['draw_ids'], core['draw_logps']
    if (any(type(token) is not int or token < 0 for token in inputs+targets+draws) or
            type(core['sampled_tokens']) is not int or core['sampled_tokens'] != len(draws) or
            not 1 <= len(draws) <= maximum or len(logps) != len(draws)):
        raise ValueError('Invalid recorded native draw sequence')
    ended = draws[-1] == eos and len(draws) < maximum
    expected_targets = draws if ended else draws[:-1]
    expected_logps = logps if ended else logps[:-1]
    if eos in draws[:-1] or (not ended and len(draws) < maximum):
        raise ValueError('Draws continue past EOS or stop before EOS/cap')
    if (core['stop'] != ('eos' if ended else 'token_limit') or targets != expected_targets or
            inputs != ([bos]+targets[:-1] if targets else []) or
            core['decoded_token_ids'] != [bos]+draws[:-1] or
            core['sampling_token_logps'] != expected_logps or
            core['sampling_logp'] != sum(expected_logps) or
            core['temperature'] != artifact['temperature']):
        raise ValueError('Recorded target, denominator, or stop semantics do not match native draws')
    # Canonical encoding rejects NaN and infinity, including every unused draw.
    canonical_sha(logps)


def load_text_batches(folders, traces):
    """Validate full N8 text coverage and immutable replay bindings, without CUDA.

    Returns artifact-ordered member events for matched replay. Callers must also
    compare rebuilt singleton kv_lens/ropes to each member's recorded values.
    RNG sidecars are hash checked, not deserialized. No artifacts are modified.
    """
    folders = [Path(folder).resolve() for folder in folders]
    if len(folders) != 8 or len(traces) != 8 or len(set(folders)) != 8:
        raise ValueError('Paired replay loader requires one complete unique N8 group')
    group_root = folders[0].parent
    if any(folder.parent != group_root for folder in folders):
        raise ValueError('Rollout folders belong to different question groups')
    info = json.loads((group_root/'group.json').read_text())
    experiment = info['configuration']
    if (info['status'] != 'complete' or info['completed_rollouts'] != 8 or
            info['experimental_configuration_digest'] != canonical_sha(experiment) or
            info['source_fingerprint'] != canonical_sha(experiment['source_identity']) or
            info['source_fingerprint'] != experiment['source_fingerprint'] or
            info['rollout_configuration_digest'] != experiment['rollout_configuration_digest'] or
            experiment['pairs'] != BATCH_MEMBERS or
            experiment['text_likelihood_backend'] != TEXT_LIKELIHOOD_BACKEND or
            experiment.get('rollout_execution') != ROLLOUT_EXECUTION):
        raise ValueError('Incomplete or inconsistent experimental question configuration')
    identities = ('group', 'example_id', 'policy_digest', 'prior_optimizer_updates',
                  'rollout_configuration_digest', 'experimental_configuration_digest', 'source_fingerprint')
    samples, folder_to_index, expected = {}, {}, set()
    for position, (folder, trace) in enumerate(zip(folders, traces)):
        sample = json.loads((folder/'sampling.json').read_text())
        index = sample['rollout_index']
        if type(index) is not int or index != position or index in samples:
            raise ValueError('Invalid or repeated logical rollout index')
        if any(sample.get(key) != info.get(key) for key in identities):
            raise ValueError('Sampling policy, group, or source identity mismatch')
        if (sample['status'] != 'complete' or sample['layout_revision'] != LAYOUT_REVISION or
                sample['seed'] != experiment['rollout']['seed']+info['group']*1000+index or
                sample['base_seed'] != experiment['rollout']['seed']):
            raise ValueError('Sampling seed/status/layout does not match fixed question configuration')
        for name in ('initial', 'final'):
            if file_sha(folder/f'rng_{name}.pt') != sample[f'rng_{name}_sha256']:
                raise ValueError('Rollout RNG snapshot hash mismatch')
        samples[index] = dict(sample=sample, folder=folder, trace=trace)
        folder_to_index[folder] = index
        expected.update((index, n) for n, event in enumerate(trace) if event['kind'] == 'text')
    references = {}
    for index, data in samples.items():
        for event_index, event in enumerate(data['trace']):
            if event['kind'] != 'text':
                continue
            reference = event.get('paired_replay')
            if not isinstance(reference, dict):
                raise ValueError('A text action is missing its matched replay binding')
            path = safe_child(data['folder'], reference['artifact_path'], within=group_root)
            prior = references.setdefault(path, reference['artifact_sha256'])
            if prior != reference['artifact_sha256']:
                raise ValueError('Members disagree about the artifact hash')
    result, covered, batches_by_pair = [], set(), {}
    for path, expected_hash in sorted(references.items()):
        if file_sha(path) != expected_hash:
            raise ValueError('Text batch artifact hash mismatch')
        artifact = json.loads(path.read_text())
        if (artifact['schema_version'] != SCHEMA_VERSION or artifact['kind'] != 'paired_text_batch' or
                artifact['layout_revision'] != LAYOUT_REVISION or
                artifact['layout_sha256'] != canonical_sha(layout_document(artifact)) or
                any(artifact.get(key) != info.get(key) for key in identities)):
            raise ValueError('Unsupported or inconsistently bound text artifact')
        if info.get('cross_question_sampling') or artifact.get('cross_question_sampling'):
            from .question_batching import validate_projection
            validate_projection(group_root, info, artifact)
        count, pair = len(artifact['members']), artifact['pair_index']
        if type(pair) is not int or pair not in range(len(BATCH_MEMBERS)) or not 1 <= count <= TEXT_BATCH_SIZE or count != artifact['member_count']:
            raise ValueError('Invalid paired/singleton artifact membership')
        batch_index = artifact['batch_index']
        if (type(batch_index) is not int or batch_index < 0 or
                artifact['batch_id'] != f'g{info["group"]:06d}_p{pair:02d}_b{batch_index:03d}' or
                path != group_root/f'pair_{pair:02d}'/f'batch_{batch_index:03d}.json'):
            raise ValueError('Text batch identity/path does not match its declared pair order')
        if (artifact['max_length'] != experiment['rollout']['native_settings']['max_think_token_n'] or
                artifact['temperature'] != experiment['rollout']['native_settings']['text_temperature'] or
                artifact['bos_id'] != info['special_token_ids']['bos_token_id'] or
                artifact['eos_id'] != info['special_token_ids']['eos_token_id']):
            raise ValueError('Artifact changed native generation settings')
        sidecar = safe_child(path.parent, artifact['rng_sidecar']['file'], within=group_root)
        if file_sha(sidecar) != artifact['rng_sidecar']['sha256']:
            raise ValueError('Text batch RNG sidecar hash mismatch')
        members, rows = [], []
        for row, member in enumerate(artifact['members']):
            folder = safe_child(group_root, member['folder'])
            index = folder_to_index.get(folder)
            if (member['row'] != row or index != member['rollout_index'] or
                    index not in BATCH_MEMBERS[pair] or index in rows):
                raise ValueError('Artifact row/rollout/pair identity mismatch')
            rows.append(index)
            data = samples[index]
            event_index = member['trace_event_index']
            if type(event_index) is not int or not 0 <= event_index < len(data['trace']):
                raise ValueError('Artifact points to an invalid trace event')
            event = data['trace'][event_index]
            core = event_core(event)
            reference = event.get('paired_replay', {})
            if (event['kind'] != 'text' or member['event_core'] != core or
                    member['event_core_sha256'] != canonical_sha(core) or
                    reference.get('batch_id') != artifact['batch_id'] or reference.get('row') != row or
                    reference.get('artifact_sha256') != expected_hash or
                    safe_child(folder, reference['artifact_path'], within=group_root) != path or
                    member['seed'] != data['sample']['seed']):
                raise ValueError('Trace core or denominator is not bound to the declared text batch')
            action_index = sum(e['kind'] == 'text' for e in data['trace'][:event_index])
            if member['text_action_index'] != action_index:
                raise ValueError('Text action index differs from event order')
            if member['prefix_assets'] != prefix_assets(folder, core['prefix']):
                raise ValueError('Recorded conditioning image/VAE asset hash mismatch')
            if (len(member['prefix_kv_lens']) != 1 or len(member['prefix_ropes']) != 1 or
                    any(type(n) is not int or n < 0 for n in member['prefix_kv_lens']+member['prefix_ropes'])):
                raise ValueError('Invalid singleton prefix layout')
            validate_event(core, artifact)
            key = (index, event_index)
            if key in covered:
                raise ValueError('Text action covered by more than one batch')
            covered.add(key)
            members.append(dict(rollout_index=index, text_action_index=action_index,
                trace_event_index=event_index, event=event, folder=folder,
                prefix_kv_lens=member['prefix_kv_lens'], prefix_ropes=member['prefix_ropes']))
        draw_counts = [len(m['event']['draw_ids']) for m in members]
        schedule = [[row for row, count in enumerate(draw_counts) if step < count]
                    for step in range(max(draw_counts))]
        if (artifact['active_lane_schedule'] != schedule or
                artifact['model_forward_count'] != len(schedule) or
                artifact['total_sampled_tokens'] != sum(draw_counts)):
            raise ValueError('Active-lane schedule drops a scored or cap-only draw')
        batches_by_pair.setdefault(pair, []).append((artifact['batch_index'], rows))
        result.append(dict(batch_id=artifact['batch_id'], layout_revision=artifact['layout_revision'],
            bos_id=artifact['bos_id'], eos_id=artifact['eos_id'],
            active_lane_schedule=artifact['active_lane_schedule'], members=members,
            artifact_path=path, artifact_sha256=expected_hash,
            policy_digest=artifact['policy_digest']))
    if covered != expected or set(batches_by_pair) != set(range(len(BATCH_MEMBERS))):
        raise ValueError('Full N8 text actions are not covered exactly once')
    for pair, batches in batches_by_pair.items():
        batches.sort()
        if [index for index, rows in batches] != list(range(len(batches))) or batches[0][1] != BATCH_MEMBERS[pair]:
            raise ValueError('Pair batch order or initial batch membership is invalid')
        previous = batches[0][1]
        for index, rows in batches[1:]:
            if rows != [n for n in previous if n in rows]:
                raise ValueError('A finished lane reappears or lane order changes across text batches')
            previous = rows
    return result
