"""CPU contracts for lane/cache/RNG isolation; no CUDA parity claim."""
from copy import deepcopy
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from thinkmorph_grpo.paired_sampling import LaneRandomStream, sample_paired_text


BOS, EOS, VOCAB = 0, 5, 6


class Cache:
    def __init__(self, values):
        self.key_cache = {0: torch.tensor(values, dtype=torch.float64).reshape(-1, 1, 1)}
        self.value_cache = {0: self.key_cache[0].clone()}


def prior(marker, length):
    values = [float(marker)] + [marker + index / 10 for index in range(1, length)]
    return dict(kv_lens=[length], ropes=[0], past_key_values=Cache(values))


def reference_logits(marker, position, full_history, eos_at):
    if position == eos_at.get(marker):
        output = torch.full((VOCAB,), -torch.inf, dtype=torch.float64)
        output[EOS] = 0
        return output
    scale = sum(full_history) / 300
    output = torch.tensor([-.4, .2, -.1, .6, -.8, -torch.inf], dtype=torch.float64)
    output[:EOS] += torch.arange(EOS, dtype=torch.float64) * scale
    return output


class FakeNative(nn.Module):
    """A tiny native-shaped model whose logits depend on its own packed KV block."""
    def __init__(self, eos_at):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.eos_at = eos_at
        self.use_moe = True
        self.forward_records = []
        self.language_model = SimpleNamespace(model=SimpleNamespace(embed_tokens=self.embed),
            forward_inference=self.forward_inference, lm_head=lambda values: values)
        self.eval()

    @staticmethod
    def embed(tokens):
        return tokens.to(torch.float64).reshape(-1, 1)

    @staticmethod
    def prepare_start_tokens(lengths, ropes, ids):
        return dict(packed_start_tokens=torch.full((len(lengths),), ids['bos_token_id'], dtype=torch.long),
            packed_query_position_ids=torch.tensor(ropes, dtype=torch.long),
            key_values_lens=torch.tensor(lengths, dtype=torch.int32))

    def forward_inference(self, *, packed_query_sequence, query_lens, packed_query_position_ids,
                          packed_query_indexes, past_key_values, key_values_lens,
                          packed_key_value_indexes, update_past_key_values, is_causal, mode):
        assert mode == 'und' and is_causal and update_past_key_values
        assert query_lens.tolist() == [1] * len(key_values_lens)
        old = past_key_values.key_cache[0]
        merged = torch.empty((old.shape[0] + len(query_lens), 1, 1), dtype=torch.float64)
        assert sorted(packed_query_indexes.tolist() + packed_key_value_indexes.tolist()) == list(range(len(merged)))
        merged[packed_key_value_indexes] = old
        merged[packed_query_indexes] = packed_query_sequence.reshape(-1, 1, 1)
        parts = merged.split((key_values_lens + 1).tolist())
        logits, markers = [], []
        for row, block in enumerate(parts):
            marker = int(block[0].item())
            markers.append(marker)
            logits.append(reference_logits(marker, int(packed_query_position_ids[row]),
                                           block.flatten().tolist(), self.eos_at))
        self.forward_records.append(dict(markers=markers, lengths=key_values_lens.tolist(),
                                         tokens=packed_query_sequence.flatten().tolist()))
        past_key_values.key_cache[0] = merged
        past_key_values.value_cache[0] = merged.clone()
        return SimpleNamespace(past_key_values=past_key_values, packed_query_sequence=torch.stack(logits))


def independent_reference(context, *, seed, eos_at, max_length, temperature=1.):
    generator = torch.Generator().manual_seed(seed)
    original = context['past_key_values'].key_cache[0].flatten().tolist()
    marker = int(original[0])
    full_history, draws, logps = list(original), [], []
    current = BOS
    for step in range(max_length):
        full_history.append(current)
        raw = torch.softmax(reference_logits(marker, step, full_history, eos_at) / temperature, dim=-1)
        draw = torch.multinomial(raw.unsqueeze(0), 1, generator=generator)
        current = int(draw.item())
        distribution = raw.double() / raw.double().sum()
        draws.append(current)
        logps.append(float(distribution[current].log()))
        if current == EOS:
            break
    return draws, logps, generator.get_state()


class PairedGenerationTests(unittest.TestCase):
    def run_pair(self, eos_at, max_length=7, seeds=(17, 29), contexts=None):
        contexts = contexts or [prior(100, 2), prior(200, 5)]
        generators = [torch.Generator().manual_seed(seed) for seed in seeds]
        model = FakeNative(eos_at)
        result = sample_paired_text(model, contexts, bos_id=BOS, eos_id=EOS,
            max_length=max_length, temperature=1., generators=generators)
        return result, model, generators, contexts

    def test_variable_prefixes_eos_compaction_and_actual_per_lane_likelihoods(self):
        eos_at = {100: 1, 200: 4}
        result, model, generators, contexts = self.run_pair(eos_at)
        self.assertEqual(result['active_lane_schedule'], [[0, 1], [0, 1], [1], [1], [1]])
        self.assertEqual([x['markers'] for x in model.forward_records], [[100,200], [100,200], [200], [200], [200]])
        for lane, (context, seed) in enumerate(zip(contexts, (17,29))):
            draws, logs, state = independent_reference(context, seed=seed, eos_at=eos_at, max_length=7)
            actual = result['lanes'][lane]
            self.assertEqual(actual['draw_ids'], draws)
            self.assertEqual(actual['target_ids'], draws)
            self.assertEqual(actual['input_ids'], [BOS]+draws[:-1])
            self.assertEqual(actual['decoded_token_ids'], [BOS]+draws[:-1])
            torch.testing.assert_close(torch.tensor(actual['sampling_token_logps']), torch.tensor(logs), rtol=0, atol=0)
            self.assertEqual(actual['sampling_logp'], sum(logs))
            self.assertTrue(torch.equal(generators[lane].get_state(), state))
        self.assertEqual(result['model_forward_count'], 5)
        self.assertEqual(result['total_sampled_tokens'], 7)

    def test_cap_discards_only_last_likelihood_but_keeps_draw_and_forward(self):
        result, _, generators, contexts = self.run_pair({100: 0, 200: 2}, max_length=3)
        self.assertEqual(result['active_lane_schedule'], [[0,1], [1], [1]])
        short, capped = result['lanes']
        self.assertEqual((short['stop'], short['sampled_tokens'], short['target_ids']), ('eos', 1, [EOS]))
        self.assertEqual((capped['stop'], capped['sampled_tokens']), ('token_limit', 3))
        self.assertEqual(capped['draw_ids'][-1], EOS)
        self.assertEqual(capped['target_ids'], capped['draw_ids'][:-1])
        self.assertEqual(capped['sampling_token_logps'], capped['draw_logps'][:-1])
        _, _, state = independent_reference(contexts[1], seed=29, eos_at={100:0,200:2}, max_length=3)
        self.assertTrue(torch.equal(generators[1].get_state(), state))

    def test_reverse_completion_order_and_lane_permutation_preserve_ownership(self):
        eos_at = {100:4,200:1}
        original, _, original_generators, _ = self.run_pair(eos_at)
        reversed_pair, _, reversed_generators, _ = self.run_pair(eos_at,
            contexts=[prior(200,5),prior(100,2)], seeds=(29,17))
        self.assertEqual(original['active_lane_schedule'], [[0,1],[0,1],[0],[0],[0]])
        self.assertEqual(reversed_pair['active_lane_schedule'], [[0,1],[0,1],[1],[1],[1]])
        for left,right,left_rng,right_rng in zip(original['lanes'],reversed(reversed_pair['lanes']),
                original_generators,reversed(reversed_generators)):
            self.assertEqual(left['draw_ids'],right['draw_ids'])
            self.assertEqual(left['sampling_token_logps'],right['sampling_token_logps'])
            self.assertTrue(torch.equal(left_rng.get_state(),right_rng.get_state()))

    def test_bfloat16_probabilities_are_renormalized_in_double_before_recording(self):
        model=FakeNative({})
        logits=torch.tensor([-.4,.2,-.1,.6,-.8,-torch.inf],dtype=torch.bfloat16)
        model.language_model.lm_head=lambda values: logits.repeat(values.shape[0],1)
        result=sample_paired_text(model,[prior(100,2),prior(200,5)],bos_id=BOS,eos_id=EOS,
            max_length=3,temperature=1.,generators=[torch.Generator().manual_seed(17),torch.Generator().manual_seed(29)])
        raw=torch.softmax(logits,dim=-1)
        self.assertNotEqual(float(raw.double().sum()),1.)
        normalized=raw.double()/raw.double().sum()
        for lane in result['lanes']:
            expected=[float(normalized[token].log()) for token in lane['draw_ids']]
            self.assertEqual(lane['draw_logps'],expected)
            self.assertEqual(lane['sampling_logp'],sum(expected[:-1]))

    def test_two_capped_lanes_and_empty_retained_cap_one_action(self):
        result, _, _, _ = self.run_pair({}, max_length=4)
        self.assertEqual(result['active_lane_schedule'], [[0,1]]*4)
        self.assertTrue(all(lane['sampled_tokens']==4 and len(lane['target_ids'])==3 for lane in result['lanes']))
        result, _, _, _ = self.run_pair({100:0}, max_length=1)
        for lane in result['lanes']:
            self.assertEqual(lane['target_ids'], [])
            self.assertEqual(lane['input_ids'], [])
            self.assertEqual(lane['decoded_token_ids'], [BOS])
            self.assertEqual(lane['sampling_logp'], 0)
            self.assertEqual(lane['stop'], 'token_limit')

    def test_sampling_does_not_mutate_original_prefix_or_process_rng(self):
        contexts = [prior(100,2), prior(200,5)]
        originals = [deepcopy(context) for context in contexts]
        model = FakeNative({100:2,200:4})
        generators = [torch.Generator().manual_seed(seed) for seed in (17,29)]
        before = torch.get_rng_state().clone()
        sample_paired_text(model, contexts, bos_id=BOS, eos_id=EOS,
            max_length=7, temperature=1., generators=generators)
        self.assertTrue(torch.equal(torch.get_rng_state(), before))
        for current, original in zip(contexts, originals):
            self.assertEqual(current['kv_lens'], original['kv_lens'])
            self.assertEqual(current['ropes'], original['ropes'])
            for field in ('key_cache','value_cache'):
                self.assertTrue(torch.equal(getattr(current['past_key_values'],field)[0],
                                            getattr(original['past_key_values'],field)[0]))

    def test_one_surviving_lane_matches_independent_reference(self):
        result, _, generators, contexts = self.run_pair({200:3}, contexts=[prior(200,5)], seeds=(29,))
        draws, logs, state = independent_reference(contexts[0], seed=29, eos_at={200:3}, max_length=7)
        self.assertEqual(result['active_lane_schedule'], [[0]]*4)
        self.assertEqual(result['lanes'][0]['draw_ids'], draws)
        self.assertEqual(result['lanes'][0]['sampling_logp'], sum(logs))
        self.assertTrue(torch.equal(generators[0].get_state(), state))

    def test_shared_rng_training_mode_and_invalid_settings_are_rejected(self):
        model = FakeNative({})
        contexts = [prior(100,2),prior(200,5)]
        generator = torch.Generator().manual_seed(17)
        settings = dict(bos_id=BOS,eos_id=EOS,max_length=4,temperature=1.,generators=[generator,generator])
        with self.assertRaisesRegex(ValueError,'independent'):
            sample_paired_text(model,contexts,**settings)
        settings['generators'] = [generator,torch.Generator().manual_seed(29)]
        for change in (dict(temperature=0),dict(temperature=float('nan')),dict(max_length=0),dict(bos_id=True)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                sample_paired_text(model,contexts,**{**settings,**change})
        model.train()
        with self.assertRaisesRegex(ValueError,'evaluation'):
            sample_paired_text(model,contexts,**settings)


class LaneRandomStreamTests(unittest.TestCase):
    def setUp(self):
        self.cpu_only = patch('torch.cuda.is_available',return_value=False)
        self.cpu_only.start()
        self.addCleanup(self.cpu_only.stop)

    @staticmethod
    def random_operation():
        return random.random(), float(np.random.random()), torch.randn(5)

    def assert_random_equal(self,left,right):
        self.assertEqual(left[:2],right[:2])
        self.assertTrue(torch.equal(left[2],right[2]))

    def test_interleaved_prefix_text_and_image_streams_equal_independent_rollouts(self):
        from thinkmorph_grpo.randomness import seed_everything, capture_rng_state, restore_rng_state
        caller = capture_rng_state()
        seeds = (17,29)
        expected=[]
        for seed in seeds:
            seed_everything(seed)
            prefix=self.random_operation()
            torch.multinomial(torch.tensor([[.1,.2,.7]]),1)
            torch.multinomial(torch.tensor([[.6,.3,.1]]),1)
            image=self.random_operation()
            expected.append((prefix,image,capture_rng_state()))
        restore_rng_state(caller)
        streams=[LaneRandomStream.from_seed(seed) for seed in seeds]
        prefixes=[]
        for stream in streams:
            with stream.scope(): prefixes.append(self.random_operation())
        generators=[stream.generator('cpu') for stream in streams]
        for distribution in (torch.tensor([[.1,.2,.7]]),torch.tensor([[.6,.3,.1]])):
            for generator in generators: torch.multinomial(distribution,1,generator=generator)
        images=[]
        for stream,generator in zip(streams,generators):
            stream.adopt_generator(generator)
            with stream.scope(): images.append(self.random_operation())
        for index in range(2):
            self.assert_random_equal(prefixes[index],expected[index][0])
            self.assert_random_equal(images[index],expected[index][1])
            self.assertTrue(torch.equal(streams[index].snapshot()['torch'],expected[index][2]['torch']))
        actual=capture_rng_state()
        self.assertEqual(actual['python'],caller['python'])
        self.assertTrue(torch.equal(actual['numpy']['keys'],caller['numpy']['keys']))
        for field in ('name','position','has_gauss','cached_gauss'):
            self.assertEqual(actual['numpy'][field],caller['numpy'][field])
        self.assertTrue(torch.equal(actual['torch'],caller['torch']))

    def test_scope_failure_preserves_consumption_and_restores_process(self):
        from thinkmorph_grpo.randomness import capture_rng_state
        caller=capture_rng_state()
        stream=LaneRandomStream.from_seed(42)
        initial=stream.snapshot()['torch']
        with self.assertRaisesRegex(RuntimeError,'operation failed'):
            with stream.scope():
                torch.randn(3)
                raise RuntimeError('operation failed')
        self.assertFalse(torch.equal(initial,stream.snapshot()['torch']))
        self.assertTrue(torch.equal(caller['torch'],capture_rng_state()['torch']))
        self.assertFalse(stream.active)


if __name__ == '__main__':
    unittest.main(verbosity=2)
