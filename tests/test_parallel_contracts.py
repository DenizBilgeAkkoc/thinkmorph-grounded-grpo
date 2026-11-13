"""CPU contracts for experimental parallel likelihoods and snapshot guards."""
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'src'))
from thinkmorph_grpo.parallel_replay import parallel_token_logprobs, snapshot_parallel_token_logprobs
from thinkmorph_grpo.parallel_trainer import ParallelGRPOTrainer, _verify_parallel_old_new
from thinkmorph_grpo import parallel_trainer


class ToyLanguage(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(32, 8)
        self.model.layers = torch.nn.ModuleList([])
        self.lm_head = torch.nn.Linear(8, 32)
        self.calls = []

    def forward_inference(self, **settings):
        self.calls.append(settings)
        assert not settings['update_past_key_values'] and settings['is_causal']
        h = settings['packed_query_sequence']
        cache = settings['past_key_values']
        qlens, klens = settings['query_lens'].tolist(), settings['key_values_lens'].tolist()
        merged = h.new_zeros((sum(qlens)+sum(klens), 8))
        merged[settings['packed_key_value_indexes']] = cache.value_cache[0][:, 0]
        merged[settings['packed_query_indexes']] = h
        result, offset, row = [], 0, 0
        for nq, nk in zip(qlens, klens):
            for i in range(nq):
                result.append(h[row] + merged[offset:offset+nk+i+1].mean(0)
                              + settings['packed_query_position_ids'][row].float()/100)
                row += 1
            offset += nq+nk
        return SimpleNamespace(packed_query_sequence=torch.stack(result))


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = ToyLanguage()
        self.use_moe = True
        self.eval()


def event(targets, stop='eos', bos=1, temperature=.8):
    draws = targets + ([11] if stop == 'token_limit' else [])
    return dict(input_ids=[bos]+targets[:-1] if targets else [], target_ids=targets,
        stop=stop, sampled_tokens=len(draws), draw_ids=draws, temperature=temperature)


def schedule(events):
    return [[lane for lane,e in enumerate(events) if t<e['sampled_tokens']]
            for t in range(max(e['sampled_tokens'] for e in events))]


def contexts():
    return [dict(kv_lens=[length], ropes=[position],
        past_key_values=SimpleNamespace(key_cache={0:torch.randn(length,1,8,requires_grad=True)},
            value_cache={0:torch.randn(length,1,8,requires_grad=True)}))
        for length,position in [(2,7),(3,13)]]


def manual_scores(model, priors, events):
    result=[]
    for prior,e in zip(priors,events):
        rows=[]
        for i,token in enumerate(e['input_ids']):
            h=model.language_model.model.embed_tokens(torch.tensor(e['input_ids'][:i+1]))
            prefix=prior['past_key_values'].value_cache[0].detach()[:,0]
            value=h[-1]+torch.cat([prefix,h]).mean(0)+(prior['ropes'][0]+i)/100
            logits=model.language_model.lm_head(value).float()/e['temperature']
            rows.append(F.log_softmax(logits,dim=-1)[e['target_ids'][i]].double())
        result.append(torch.stack(rows) if rows else torch.empty(0,dtype=torch.float64))
    return result


class ParallelContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.model, self.priors = ToyModel(), contexts()

    def test_eos_cap_packing_and_scores_against_independent_token_oracle(self):
        events=[event([2,3,4]),event([6],stop='token_limit')]
        actual=parallel_token_logprobs(self.model,self.priors,events,active_lane_schedule=schedule(events),bos_id=1,checkpoint_chunk_size=2)
        expected=manual_scores(self.model,self.priors,events)
        for a,b in zip(actual,expected):
            torch.testing.assert_close(a,b,rtol=0,atol=5e-7)
        settings=self.model.language_model.calls[-1]
        self.assertEqual(settings['query_lens'].tolist(),[3,1])
        self.assertEqual(settings['packed_query_position_ids'].tolist(),[7,8,9,13])
        self.assertEqual(settings['packed_query_indexes'].tolist(),[2,3,4,8])
        self.assertEqual(settings['packed_key_value_indexes'].tolist(),[0,1,5,6,7])
        sum(x.sum() for x in actual).backward()
        self.assertGreater(float(self.model.language_model.model.embed_tokens.weight.grad.abs().sum()),0)
        self.assertTrue(all(p['past_key_values'].value_cache[0].grad is None for p in self.priors))

    def test_old_new_identity_detachment_prefix_and_rng(self):
        events=[event([2,3,4]),event([6,7])]
        before=torch.get_rng_state().clone()
        saved=[p['past_key_values'].value_cache[0].detach().clone() for p in self.priors]
        old=snapshot_parallel_token_logprobs(self.model,self.priors,events,active_lane_schedule=schedule(events),bos_id=1,checkpoint_chunk_size=2)
        current=parallel_token_logprobs(self.model,self.priors,events,active_lane_schedule=schedule(events),bos_id=1,checkpoint_chunk_size=2)
        checks=_verify_parallel_old_new(current,old,tolerance=1e-5)
        self.assertTrue(all(c['unchanged_policy_ratio']==1 for c in checks))
        self.assertTrue(all(not x.requires_grad for x in old))
        for p,v in zip(self.priors,saved):
            torch.testing.assert_close(p['past_key_values'].value_cache[0],v,rtol=0,atol=0)
        torch.testing.assert_close(torch.get_rng_state(),before,rtol=0,atol=0)
        with self.assertRaisesRegex(ValueError,'old/new unchanged-policy mismatch'):
            _verify_parallel_old_new([current[0]+.1,current[1]],old,tolerance=1e-5)

    def test_zero_scored_lane_and_all_empty_caps(self):
        events=[event([],stop='token_limit'),event([4,5])]
        scores=parallel_token_logprobs(self.model,self.priors,events,active_lane_schedule=schedule(events),bos_id=1)
        self.assertEqual([x.numel() for x in scores],[0,2])
        self.assertEqual(self.model.language_model.calls[-1]['query_lens'].tolist(),[2])
        self.assertEqual(self.model.language_model.calls[-1]['packed_query_position_ids'].tolist(),[13,14])
        empty=[event([],stop='token_limit'),event([],stop='token_limit')]
        calls=len(self.model.language_model.calls)
        result=parallel_token_logprobs(self.model,self.priors,empty,active_lane_schedule=schedule(empty),bos_id=1)
        self.assertEqual([x.numel() for x in result],[0,0])
        self.assertEqual(len(self.model.language_model.calls),calls)
        sum(x.sum() for x in result).backward()

    def test_invalid_archive_schedule_shift_and_temperature_rejected(self):
        events=[event([2,3]),event([6])]
        with self.assertRaisesRegex(ValueError,'Active-lane schedule'):
            parallel_token_logprobs(self.model,self.priors,events,active_lane_schedule=[[0,1]],bos_id=1)
        changed=deepcopy(events);changed[0]['input_ids'][1]=9
        with self.assertRaisesRegex(ValueError,'input/target shift'):
            parallel_token_logprobs(self.model,self.priors,changed,active_lane_schedule=schedule(changed),bos_id=1)
        changed=deepcopy(events);changed[1]['temperature']=1.1
        with self.assertRaisesRegex(ValueError,'shared scalar temperature'):
            parallel_token_logprobs(self.model,self.priors,changed,active_lane_schedule=schedule(changed),bos_id=1)

    def test_old_new_guard_rejects_a_differentiable_old_score(self):
        with self.assertRaisesRegex(ValueError,'Invalid parallel old/new'):
            _verify_parallel_old_new([torch.ones(1,dtype=torch.float64)],
                [torch.ones(1,dtype=torch.float64,requires_grad=True)],tolerance=1e-5)

    def test_archived_policy_guard_precedes_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            folders=[]
            for i in range(8):
                folder=Path(directory)/str(i);folder.mkdir();folders.append(folder)
                (folder/'sampling.json').write_text(json.dumps(dict(rollout_index=i,policy_digest='other')))
            trainer=ParallelGRPOTrainer(SimpleNamespace(model=self.model),None,None)
            with patch.object(parallel_trainer.native,'adapter_digest',return_value='current'):
                with self.assertRaisesRegex(ValueError,'Rollout policy differs'):
                    trainer._update(folders,[{}]*8,optimize=True)

    def test_image_update_ast_is_unchanged(self):
        package=ROOT.parent/'snapshots/production/thinkmorph_grpo'
        def image_loop(path):
            tree=ast.parse(path.read_text())
            update=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_update')
            return next(n for n in update.body if isinstance(n,ast.For) and
                        isinstance(n.target,ast.Tuple) and len(n.target.elts)==2 and
                        isinstance(n.target.elts[1],ast.Tuple) and
                        [x.id for x in n.target.elts[1].elts]==['folder','steps'])
        self.assertEqual(ast.dump(image_loop(package/'paired_trainer.py')),
                         ast.dump(image_loop(package/'parallel_trainer.py')))


if __name__=='__main__':
    unittest.main(verbosity=2)
