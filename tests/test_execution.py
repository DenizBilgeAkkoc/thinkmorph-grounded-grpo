import sys,unittest
from pathlib import Path
from copy import deepcopy
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from thinkmorph_grpo.execution_tuning import replay_batches

class ReplayLayout(unittest.TestCase):
    def test_wider_generation_recovers_original_logical_pairs(self):
        original=[];wide=[]
        for action in range(2):
            members=[dict(rollout_index=n,text_action_index=action,event=dict(sampled_tokens=3+n%3)) for n in range(8) if action==0 or n!=3]
            wide.append(dict(batch_id='gen8_'+str(action),members=members))
        for pair in range(4):
            for action in range(2):
                original.append([(n,action) for n in range(2*pair,2*pair+2) if action==0 or n!=3])
        before=deepcopy(wide);actual=replay_batches(wide)
        self.assertEqual([[(m['rollout_index'],m['text_action_index']) for m in b['members']] for b in actual],original)
        self.assertEqual(wide,before)
        for b in actual:
            counts=[m['event']['sampled_tokens'] for m in b['members']]
            self.assertEqual(b['active_lane_schedule'],[[i for i,c in enumerate(counts) if s<c] for s in range(max(counts))])
    def test_duplicate_actions_rejected(self):
        member=dict(rollout_index=0,text_action_index=0,event=dict(sampled_tokens=3))
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            replay_batches([dict(batch_id='bad',members=[member,member])])

if __name__=='__main__':unittest.main(verbosity=2)
