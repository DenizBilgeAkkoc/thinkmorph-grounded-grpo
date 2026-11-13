"""Runner contracts that must hold before spending GPU time on native models."""
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from thinkmorph_grpo import train as base
from thinkmorph_grpo.bridge import atomic_json
from thinkmorph_grpo.judge_v2 import JudgeAssessmentError
from thinkmorph_grpo.train_distributed import (
    Coordinator, DistributedPhaseError, UnscorableJudgeGroup,
    _checkpoint_paths, _restore, _sample_and_judge, _save, pin_visible_gpu,
)


class ScriptedCollective:
    """Exercise Coordinator serialization with one explicit peer response."""
    def __init__(self, rank=0, remote_error=None):
        self.rank, self.remote_error = rank, remote_error
        self.calls = []

    def all_gather_object(self, output, value, *, group):
        self.calls.append(value)
        output[self.rank] = value
        output[1-self.rank] = {"rank": 1-self.rank, "error": self.remote_error}


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def coordinator(self, rank=0, remote_error=None):
        result = Coordinator(self.root, rank, 2, object(),
                             ScriptedCollective(rank, remote_error), atomic_json)
        result.ready = True
        result.summary = {"status": "running", "groups": []}
        return result

    def test_first_status_creates_rank_directory_using_real_atomic_writer(self):
        coordinator = self.coordinator()
        self.assertFalse((self.root / 'ranks').exists())
        coordinator.status('loading_model', group=0)
        status = json.loads((self.root/'ranks/rank_00000/status.json').read_text())
        self.assertEqual(status['phase'], 'loading_model')
        self.assertEqual(status['group'], 0)
        self.assertEqual(json.loads((self.root/'summary.json').read_text())['phase'], 'loading_model')

    def test_nonzero_rank_never_overwrites_shared_summary(self):
        original = {'status': 'rank_zero_owns_this', 'phase': 'sampling'}
        atomic_json(self.root/'summary.json', original)
        coordinator = self.coordinator(rank=1)
        coordinator.status('replay_backward', group=1)
        self.assertEqual(json.loads((self.root/'summary.json').read_text()), original)
        self.assertEqual(json.loads((self.root/'ranks/rank_00001/status.json').read_text())['rank'], 1)

    def test_local_error_is_shared_and_persisted_before_phase_returns(self):
        coordinator = self.coordinator()
        def fail():
            raise ValueError('sampling failed before any optimizer step')
        with self.assertRaisesRegex(DistributedPhaseError, 'rank 0 ValueError'):
            coordinator.phase('local_work', fail)
        failure = json.loads((self.root/'ranks/rank_00000/failure.json').read_text())
        self.assertEqual(failure['errors'][0]['rank'], 0)
        self.assertIn('sampling failed', failure['errors'][0]['message'])
        self.assertIn('ValueError', coordinator.dist.calls[0]['error']['traceback'])

    def test_remote_error_cannot_be_hidden_by_successful_local_work(self):
        remote = {'rank': 1, 'type': 'FloatingPointError', 'message': 'bad replay', 'traceback': 'peer trace'}
        coordinator = self.coordinator(remote_error=remote)
        called = []
        with self.assertRaisesRegex(DistributedPhaseError, 'rank 1 FloatingPointError'):
            coordinator.phase('replay', lambda: called.append('local work finished'))
        self.assertEqual(called, ['local work finished'])
        failure = json.loads((self.root/'ranks/rank_00000/failure.json').read_text())
        self.assertEqual(failure, {'phase': 'replay', 'errors': [remote]})

    def test_failure_file_survives_status_writer_failure_before_directory_creation(self):
        coordinator = self.coordinator()
        with patch.object(coordinator, 'status', side_effect=RuntimeError('status unavailable')):
            with self.assertRaisesRegex(DistributedPhaseError, 'status unavailable'):
                coordinator.phase('initializing', lambda: None)
        self.assertTrue((self.root/'ranks/rank_00000/failure.json').is_file())


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_text = torch.nn.Parameter(torch.tensor([1., -2.], dtype=torch.float64))
        self.lora_moe_gen = torch.nn.Parameter(torch.tensor([.5, 1.25], dtype=torch.float64))


class SaveCoordinator:
    rank, world = 0, 2
    def __init__(self, states):
        self.states = states
    def gather(self, _value):
        return self.states
    def phase(self, _label, function, **_kwargs):
        return function()


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cpu = patch('torch.cuda.is_available', return_value=False)
        self.cpu.start()
        self.addCleanup(self.cpu.stop)
        self.model = TinyPolicy()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=.02, weight_decay=.1)
        sum(p.square().sum() for p in self.model.parameters()).backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.configuration = {'test': 'distributed_checkpoint', 'world_size': 2}
        self.states = []
        for seed in (123, 456):
            base.seed_everything(seed)
            self.states.append(base.capture_rng_state())
        base.restore_rng_state(self.states[0])
        self.path = Path(self.tmp.name)/'checkpoint.pt'
        _save(torch, base, SaveCoordinator(self.states), self.path, self.model,
              self.optimizer, self.configuration, last_group=3, optimizer_updates=2, wave=1)
        self.identity = (base.adapter_digest(self.model), base._state_digest(self.optimizer.state_dict()))

    def restore(self, rank=0, world=2):
        return _restore(torch, base, self.path, self.model, self.optimizer,
                        self.configuration, rank, world)

    def test_resume_restores_exact_adam_and_each_ranks_own_rng(self):
        for rank in range(2):
            with self.subTest(rank=rank):
                with torch.no_grad():
                    for p in self.model.parameters():
                        p.zero_()
                self.optimizer.param_groups[0]['lr'] = 123.
                base.seed_everything(999)
                result = self.restore(rank)
                self.assertEqual((base.adapter_digest(self.model), base._state_digest(self.optimizer.state_dict())), self.identity)
                self.assertEqual(base._state_digest(base.capture_rng_state()), base._state_digest(self.states[rank]))
                self.assertEqual((result['group'], result['wave'], result['optimizer_updates']), (3, 1, 2))

    def test_corrupted_rng_sidecar_is_rejected_before_mutating_model(self):
        rng_path, _ = _checkpoint_paths(self.path)
        state = torch.load(rng_path, weights_only=True)
        state['rank_rng'][1]['torch'][0] ^= 1
        torch.save(state, rng_path)
        with torch.no_grad():
            self.model.lora_text.add_(10)
        expected = base.adapter_digest(self.model)
        with self.assertRaisesRegex(ValueError, 'RNG digest mismatch'):
            self.restore(rank=1)
        self.assertEqual(base.adapter_digest(self.model), expected)

    def test_incomplete_commit_and_changed_world_size_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'world size changed'):
            self.restore(world=3)
        _, marker_path = _checkpoint_paths(self.path)
        marker_path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.restore()


class SamplingJudgingBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.image = self.root/'problem.png'
        Image.new('RGB', (2, 2), 'white').save(self.image)
        self.example = {'task': 'unit_test', 'question': 'test', 'problem_image': {'path': str(self.image)}}
        self.cfg = dict(seed=12, native_settings={'max_rounds': 1, 'num_timesteps': 15},
                        sde_window_size=3, noise_level=1., checkpoint='test_checkpoint', native_revision='test')

    def run_group(self, *, sampler_error=None, judge_error=None, judge_errors=None, check_overlap=False,
                  sample_attempt=None):
        judge_started = threading.Event()
        statuses, judged = [], []
        class Recorder:
            events = []
            samples = 0
            def begin_recording(recorder, folder, bridge):
                recorder.folder = folder
                folder.mkdir(parents=True)
            def interleave_inference(recorder, *_args, **_kwargs):
                if sampler_error is not None:
                    raise sampler_error
                recorder.samples += 1
                if check_overlap and recorder.samples == 2:
                    if not judge_started.wait(3):
                        raise AssertionError('Judge did not start while later samples were generated')
                atomic_json(recorder.folder/'trace.json', [{'kind': 'text', 'text': 'done'}])
        recorder = Recorder()
        @contextmanager
        def bridge(*_args):
            yield object()
        def judge_group(_example, folders, _judge, **_kwargs):
            judge_started.set()
            judged.append(folders[0].name)
            rollout = int(folders[0].name.split('_')[-1])
            error = (judge_errors or {}).get(rollout, judge_error)
            if error is not None:
                raise error
            self.assertTrue((folders[0]/'sampling.json').exists())
            return [{'rollout': rollout}]
        fake_base = SimpleNamespace(shared_windows=lambda *_: [1], seed_everything=lambda _: None,
            NativeFlowBridge=bridge, atomic_json=atomic_json, _state_digest=base._state_digest,
            JudgeAssessmentError=JudgeAssessmentError, judge_group=judge_group)
        coordinator = SimpleNamespace(rank=0, status=lambda phase, **kw: statuses.append((phase, kw)))
        result = _sample_and_judge(fake_base, (object(),), recorder, object(), self.example,
            self.cfg, self.root, 0, 0, 'policy_digest', 2, True, coordinator, sample_attempt=sample_attempt)
        return result, judged, statuses

    def test_judging_overlaps_sampling_and_preserves_rollout_result_order(self):
        (folders, judgments, timings), judged, statuses = self.run_group(check_overlap=True)
        self.assertEqual(len(folders), 8)
        self.assertEqual([item['rollout'] for item in judgments], list(range(8)))
        self.assertEqual(len(judged), 8)
        self.assertTrue(timings['judge_overlapped_with_sampling'])
        self.assertGreaterEqual(timings['judge_tail_seconds'], 0)
        self.assertEqual(statuses[-1][0], 'waiting_for_judge')

    def test_only_judge_semantic_failure_is_classified_unscorable(self):
        error = JudgeAssessmentError('bounded judgment failed')
        with self.assertRaises(UnscorableJudgeGroup) as raised:
            self.run_group(judge_error=error)
        self.assertIs(raised.exception.__cause__, error)

    def test_sampler_semantic_exception_type_is_still_a_fatal_sampler_error(self):
        error = JudgeAssessmentError('native sampler failed')
        with self.assertRaises(JudgeAssessmentError) as raised:
            self.run_group(sampler_error=error)
        self.assertIs(raised.exception, error)

    def test_judge_transport_failure_is_not_unscorable(self):
        error = RuntimeError('Gemini HTTP service unavailable')
        with self.assertRaises(RuntimeError) as raised:
            self.run_group(judge_error=error)
        self.assertIs(raised.exception, error)

    def test_earlier_semantic_failure_cannot_hide_later_transport_failure(self):
        # With skip-unscorable enabled, misclassifying this mixed group would
        # silently discard an infrastructure failure and continue training.
        semantic = JudgeAssessmentError('rollout zero exhausted semantic validation')
        transport = RuntimeError('rollout one Gemini HTTP service unavailable')
        with self.assertRaises(RuntimeError) as raised:
            self.run_group(judge_errors={0: semantic, 1: transport})
        self.assertNotIsInstance(raised.exception, UnscorableJudgeGroup)
        self.assertIn('Gemini HTTP service unavailable', str(raised.exception))


class VisibilityTests(unittest.TestCase):
    def test_rank_chooses_its_gpu_before_cuda_initialization(self):
        environment = {'LOCAL_RANK': '1', 'LOCAL_WORLD_SIZE': '2', 'CUDA_VISIBLE_DEVICES': 'GPU-first,GPU-second'}
        with patch.dict('os.environ', environment):
            self.assertEqual(pin_visible_gpu(), 'GPU-second')

    def test_ambiguous_device_assignment_is_rejected(self):
        environment = {'LOCAL_RANK': '1', 'LOCAL_WORLD_SIZE': '2', 'CUDA_VISIBLE_DEVICES': 'GPU-only'}
        with patch.dict('os.environ', environment):
            with self.assertRaisesRegex(RuntimeError, 'exactly one visible GPU per local worker'):
                pin_visible_gpu()


if __name__ == '__main__':
    unittest.main()
