from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from thinkmorph_grpo.judge import StepEvidence, reward_components
from thinkmorph_grpo.judge_v2 import (GroundedQwenJudge, derive_scores, validate_claim_semantics,
    validate_visual_with_checks, validate_independent_navigation)
from thinkmorph_grpo.navigation_perception import PROMPT, validate_perception
from thinkmorph_grpo.observation_normalization import coalesce_duplicate_observation_units
from test_navigation_predicates import GRID, assertion
from test_state_blind_claims import with_semantics


def perception(cells, rows=3):
    return dict(evidence="Physical cell boundaries",grid_rows=rows,grid_cols=3,path_present=bool(cells),
        boundary_crossings=[dict(from_cell=a,to_cell=b,evidence="Visible crossing") for a,b in zip(cells,cells[1:])],
        visited_cells=cells,assessable=True,issues=[])


def attach(value,cells):
    p=perception(cells);moves=validate_perception(p)
    value['visual'].update(path_present=bool(cells),path_moves=moves)
    value['checks'].pop('observed_path',None)
    value['checks']['independent_after_navigation']=dict(perception=p,moves=moves,source='after_only_adjacent_cells')
    return value


class ResearchNavigationTests(unittest.TestCase):
    def run_process(self,cells,history=(),rows=3):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'im.png';Image.new('RGB',(12,12),'white').save(path)
            quote='The AFTER_TEXT_SENTINEL arrow is visible.'
            value=attach(with_semantics(quote),cells)
            drafts=[deepcopy(value['proposal']),deepcopy(value['checks']['state_blind_claims']),perception(cells,rows),
                deepcopy(value['visual']),deepcopy(value['observations'])]
            for draft in (drafts[0],drafts[1],drafts[4]):
                draft['claims'][0].pop('quote');draft['claims'][0]['claim_id']=0
            class Judge(GroundedQwenJudge):
                def complete(self,messages,**kwargs):
                    self.requests.append((messages,kwargs));return drafts.pop(0)
            judge=Judge('unused');judge.requests=[]
            constraints=dict(source='verified_original_image',image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),grid=deepcopy(GRID))
            evidence=StepEvidence('QUESTION_SENTINEL',str(path),str(path),str(path),'Draw a route.',quote,
                history=history,task='Spatial_Navigation',original_constraints=constraints)
            result=judge.process(evidence)
            return result,judge.requests

    def test_complete_cell_path_and_probe_request_cache_identity(self):
        result,requests=self.run_process([[2,0],[1,0],[1,1],[1,2],[2,2]])
        self.assertEqual(len(requests),5)
        self.assertTrue(result['assessment']['checks']['observed_path']['reaches_goal'])
        messages,kwargs=requests[2]
        self.assertEqual(messages[0]['content'],PROMPT)
        self.assertEqual(kwargs['max_tokens'],1800)
        self.assertEqual(messages[1]['content'][0]['text'],'AFTER image only')
        self.assertNotIn('QUESTION_SENTINEL',json.dumps(messages))
        self.assertNotIn('AFTER_TEXT_SENTINEL',json.dumps(messages))
        visual_text=requests[3][0][1]['content'][0]['text']
        self.assertNotIn('AFTER_TEXT_SENTINEL',visual_text)
        self.assertNotIn('state_blind_claims',visual_text)
        self.assertIn('independent_after_navigation',visual_text)
        reward_components(result)

    def test_reverse_direction_tail_is_not_shifted_to_original_start(self):
        result,_=self.run_process([[1,0],[2,0]])
        checks=result['assessment']['checks']
        self.assertEqual(checks['independent_after_navigation']['moves'],['D'])
        self.assertNotIn('observed_path',checks)
        self.assertIn('begins away',checks['navigation_check_scope'])

    def test_adjacent_visible_out_of_bounds_is_simulated_without_repair(self):
        result,_=self.run_process([[2,0],[3,0]])
        check=result['assessment']['checks']['observed_path']
        self.assertFalse(check['legal']);self.assertEqual(check['reason'],'off_grid')

    def test_later_steps_keep_unverified_original_start_scope(self):
        result,_=self.run_process([[2,0],[1,0]],history=('Earlier step',))
        self.assertNotIn('observed_path',result['assessment']['checks'])

    def test_no_route_produces_no_empty_path_proof(self):
        result,_=self.run_process([])
        checks=result['assessment']['checks']
        self.assertFalse(checks['independent_after_navigation']['perception']['path_present'])
        self.assertNotIn('observed_path',checks)

    def test_dimension_change_is_not_repaired_to_expected_registration(self):
        with self.assertRaisesRegex(ValueError,'registration'):
            self.run_process([[2,0],[1,0]],rows=4)

    def test_transcription_disagreement_abstains_without_overwriting(self):
        value=attach(with_semantics('The arrow is visible.'),[[2,0],[1,0],[1,1],[1,2],[2,2]])
        value['visual']['path_moves']=['U','R','D'];before=deepcopy(value)
        with self.assertRaisesRegex(ValueError,'Conflicting navigation perception'):
            validate_visual_with_checks(value['visual'],value['checks'],'grounded_v8')
        self.assertEqual(value,before)

    def test_independent_path_does_not_upgrade_visual_preservation(self):
        value=attach(with_semantics('The arrow is visible.'),[[2,0],[1,0]])
        value['visual']['objects'][0]['status']='missing'
        scores=derive_scores(value,True,'grounded_v8')
        self.assertEqual(scores['image_preservation'],0)
        self.assertEqual(scores['image_execution'],0)

    def test_unverified_route_payload_cannot_become_symbolic_proof(self):
        value=attach(with_semantics('The full route U,R,R,D is safe.',
            [assertion('safe',scope='unverified')],covered=True,support='unsupported'),[[2,0],[1,0]])
        raw=value['checks']['state_blind_claims'];raw['claims'][0]['navigation_claims'][0]['route']='U,R,R,D'
        before=deepcopy(raw)
        normalized=validate_claim_semantics(raw,normalize_unverified=True)
        self.assertEqual(raw,before)
        self.assertEqual(normalized['claims'][0]['navigation_claims'][0]['scope'],'unverified')
        self.assertEqual(normalized['claims'][0]['navigation_claims'][0]['route'],'')
        value['checks']['state_blind_claims']=normalized
        self.assertEqual(derive_scores(value,True,'grounded_v8')['observation_grounding'],0)

    def test_drawn_route_literal_contradiction_is_not_normalized(self):
        value=with_semantics('The route is safe.',[assertion('safe',route='U')])
        with self.assertRaises(ValueError):validate_claim_semantics(value['checks']['state_blind_claims'],normalize_unverified=True)

    def test_tampered_independent_moves_or_start_proof_are_rejected(self):
        value=attach(with_semantics('The arrow is visible.'),[[1,0],[2,0]])
        value['checks']['observed_path']=dict(legal=True,reaches_goal=False)
        with self.assertRaises(ValueError):derive_scores(value,True,'grounded_v8')
        value['checks'].pop('observed_path');value['checks']['independent_after_navigation']['moves']=['U']
        with self.assertRaises(ValueError):derive_scores(value,True,'grounded_v8')

    def test_duplicate_aggregation_cannot_hide_uncertainty(self):
        units=[dict(claim_id=0,text='The path is safe and reaches the goal.')]
        raw=dict(evidence='',assessable=True,issues=[],claims=[
            dict(claim_id=0,evidence='safe',kind='path_safety',support='supported'),
            dict(claim_id=0,evidence='incomplete',kind='goal_reached',support='unsupported')])
        result=coalesce_duplicate_observation_units(raw,units)
        self.assertEqual(result['claims'][0]['support'],'unsupported')
        raw['claims'][0]['support']='uncertain'
        with self.assertRaises(ValueError):coalesce_duplicate_observation_units(raw,units)


if __name__=='__main__':unittest.main()
