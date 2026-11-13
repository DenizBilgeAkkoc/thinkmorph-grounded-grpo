"""CPU checks of loss semantics and fail-closed image replay dispatch."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from thinkmorph_grpo import image_batching as ib,trainer as native,replay
from thinkmorph_grpo.objective import GRPOConfig

class ImageBatchTests(unittest.TestCase):
    def run_objective(self,width,existing=0.,bad=False):
        parameter=torch.nn.Parameter(torch.tensor(0.,dtype=torch.float64));parameter.grad=torch.tensor(existing,dtype=torch.float64)
        cfg=GRPOConfig(velocity_beta=0.)
        trainer=SimpleNamespace(model=None,infer=None,config=cfg,_progress=lambda **kw:None)
        advantages=torch.tensor([[(-1 if n%2 else 1)*(n+1)/4] for n in range(7)])
        members=[dict(rollout_index=n,step_index=0,state=dict(transitions=[dict(index=k,x=torch.ones(2),old_logp=torch.tensor(1. if bad else 0.)) for k in range(3)])) for n in range(7)]
        batches=[dict(batch_id=str(n),members=members[n:n+width]) for n in range(0,7,width)]
        metrics=dict(image_losses=[],ratios=[],velocity_mse=[],replay_timing_seconds=dict(image_replay_and_backward=0.))
        def forward(model,inputs,members,k):
            return [parameter*(m['rollout_index']+1)*(k+1) for m in members],[parameter.reshape(1) for m in members]
        with patch.object(ib,'rebuild_inputs',return_value=None),patch.object(ib,'transition_logprobs',side_effect=forward):
            ib.backward_images(trainer,batches,advantages,7,metrics,[],[],True)
        expected=existing-sum(float(advantages[n,0])*(n+1)*(k+1)/(7*3) for n in range(7) for k in range(3))
        self.assertAlmostEqual(float(parameter.grad),expected,places=9)
        self.assertEqual(len(metrics['ratios']),21)
        self.assertTrue(all(r==1. for r in metrics['ratios']))
        return float(parameter.grad)
    def test_tail_batches_keep_question_denominator(self):
        self.assertAlmostEqual(self.run_objective(1),self.run_objective(4),places=12)
    def test_image_backward_preserves_text_gradients(self):
        self.run_objective(2,existing=3.25)
    def test_bad_old_likelihood_rejected_before_backward(self):
        with self.assertRaisesRegex(ValueError,'likelihood mismatch'):self.run_objective(2,bad=True)
    def test_scalar_inputs_reject_batched_events(self):
        with self.assertRaises(ValueError):replay.flow_inputs(None,dict(batched_image_replay={'row':0}),None)
    def test_scalar_replay_rejects_batched_events(self):
        with self.assertRaises(ValueError):replay.flow_logprob(None,None,dict(batched_image_replay={'row':0}),None,None)
    def test_geometry_preserves_rng(self):
        cache=SimpleNamespace(key_cache={0:torch.zeros(2,1)},value_cache={0:torch.zeros(2,1)})
        context=dict(kv_lens=[2],ropes=[3],past_key_values=cache)
        model=SimpleNamespace(prepare_vae_latent=lambda **kw:dict(packed_init_noises=torch.randn(5)))
        infer=SimpleNamespace(new_token_ids={})
        settings=dict(cfg_text_scale=1.,cfg_img_scale=1.,cfg_renorm_min=0.,cfg_renorm_type='global')
        before=torch.get_rng_state().clone()
        ib.packed_inputs(model,infer,[context,context],[dict(settings=settings,image_shape=[32,32])]*2)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
    def test_mixed_settings_rejected(self):
        settings=dict(cfg_text_scale=1.,cfg_img_scale=1.)
        with self.assertRaisesRegex(ValueError,'settings differ'):
            ib.packed_inputs(None,None,[{},{}],[dict(settings=settings),dict(settings={**settings,'num_timesteps':15})])

if __name__=='__main__':unittest.main(verbosity=2)
