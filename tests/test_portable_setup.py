"""Portable launcher safety and artifact identity tests; no GPU or live API."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from train import committed_checkpoint
from download_assets import verify

def write(path,value):path.write_text(json.dumps(value))

class PortableSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def test_checkpoint_proof_accepts_actual_runtime_schema_and_rejects_mismatch(self):
        # Runtime proofs contain digests/world size, not group/update counters.
        checkpoint=self.root/'checkpoint.pt';checkpoint.write_bytes(b'fixture')
        Path(str(checkpoint)+'.rng.pt').write_bytes(b'fixture')
        marker=dict(group=7,wave=0,optimizer_updates=1,world_size=8,
            policy_digest='policy',optimizer_digest='adam',sample_history_digest='history')
        write(Path(str(checkpoint)+'.complete.json'),marker)
        proof=dict(reload_passed=True,optimizer_state_exact=True,per_rank_rng_restored=True,
            world_size=8,policy_digest='policy',optimizer_digest='adam')
        write(self.root/'checkpoint_validation.json',proof)
        update={k:v for k,v in marker.items() if k not in ('group','world_size')};update['last_group']=7
        write(self.root/'update.json',update)
        self.assertEqual(committed_checkpoint(checkpoint),marker)
        for key,value in [('last_group',6),('policy_digest','other'),('optimizer_updates',2)]:
            write(self.root/'update.json',{**update,key:value})
            with self.assertRaises(ValueError):committed_checkpoint(checkpoint)
        write(self.root/'update.json',update)
        write(self.root/'checkpoint_validation.json',{**proof,'per_rank_rng_restored':False})
        with self.assertRaises(ValueError):committed_checkpoint(checkpoint)

    def test_prepare_profiles_freeze_inputs_without_serializing_credentials(self):
        image=self.root/'image.png';image.write_bytes(b'fixture image bytes')
        manifest=self.root/'manifest.json'
        write(manifest,[dict(problem_image=dict(path=str(image),sha256=hashlib.sha256(image.read_bytes()).hexdigest()))]*4)
        env={**os.environ,'GEMINI_API_KEY':'SECRET_SENTINEL_NOT_FOR_SERIALIZATION'}
        for profile,text_width,image_width in [('production','2','1'),('optimized','8','2')]:
            run=self.root/profile
            args=[sys.executable,str(ROOT/'scripts/train.py'),'--run-dir',str(run),'--manifest',str(manifest),
                '--profile',profile,'--gpus','2','--questions','4','--prepare-only']
            result=subprocess.run(args,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            config=json.loads((run/'Prepared_Command.json').read_text())
            self.assertEqual(config['environment']['THINKMORPH_TEXT_BATCH_SIZE'],text_width)
            self.assertEqual(config['environment']['THINKMORPH_IMAGE_BATCH_SIZE'],image_width)
            self.assertIn('--verify-checkpoint-reload',config['command'])
            for p in run.rglob('*.json'):self.assertNotIn(env['GEMINI_API_KEY'],p.read_text())
            result=subprocess.run(args+['--judge-workers','1'],env=env,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Frozen run input differs',result.stderr)

    def test_huggingface_small_file_uses_git_blob_hash(self):
        path=self.root/'small.json';raw=b'{"a":1}\n';path.write_bytes(raw)
        item=dict(bytes=len(raw),algorithm='sha1',digest=hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest())
        verify(path,item)
        with self.assertRaises(ValueError):verify(path,{**item,'digest':hashlib.sha1(raw).hexdigest()})

    def test_all_training_sources_match_archived_snapshot(self):
        inventory=json.loads((ROOT/'provenance/source_snapshots.json').read_text())
        self.assertEqual(len(inventory),64)
        for relative,item in inventory.items():
            self.assertEqual(hashlib.sha256((ROOT/relative).read_bytes()).hexdigest(),item['sha256'],relative)

    def test_fixed_selection_is_disjoint_and_balanced(self):
        from collections import Counter
        rows=json.loads((ROOT/'manifests/training_2000.selection.json').read_text())
        self.assertEqual(len(rows),2000)
        self.assertEqual(len({r['input_sha256'] for r in rows}),2000)
        for half in (rows[:1000],rows[1000:]):
            self.assertEqual(set(Counter(r['task'] for r in half).values()),{250})
        self.assertTrue(all(r['split']=='train' and 'answer' not in r and 'path' not in r['problem_image'] for r in rows))

    def test_wandb_configuration_uses_actual_run_targets_and_profile(self):
        sys.path.insert(0,str(ROOT/'logging'))
        from wandb_sidecar import Sidecar
        (self.root/'config').mkdir();phase=self.root/'phase_000000';phase.mkdir()
        manifest=self.root/'config/manifest.json';write(manifest,[{}, {}, {}, {}])
        write(self.root/'config/run_contract.json',dict(profile='optimized'))
        write(self.root/'Controller_Status.json',dict(target_questions=2))
        write(self.root/'Segments.json',dict(segments=[dict(path=str(phase),start_group=0,stop_group=2)]))
        key=self.root/'credential';key.write_text('unit-test-only')
        captured={}
        class SDK:
            Settings=SimpleNamespace
            @staticmethod
            def init(**kwargs):
                captured.update(kwargs)
                return SimpleNamespace(settings=SimpleNamespace(mode='online'),url='https://wandb.invalid/test',
                    step=0,define_metric=lambda *a,**k:None)
        args=SimpleNamespace(run_root=self.root,manifest=manifest,key_file=key,project='test',entity='test',
            run_id=None,name='test',image_side=640,input_side=256,max_generated_images=64,text_limit=12000)
        logger=Sidecar(args,wandb_module=SDK)
        try:
            logger.connect()
            self.assertTrue(captured['config']['fresh_training_run'])
            self.assertEqual(captured['config']['target_questions'],2)
            self.assertEqual(captured['config']['execution_profile'],'optimized')
            self.assertEqual(captured['mode'],'online')
            self.assertNotIn('unit-test-only',json.dumps(captured['config']))
        finally:logger.lock.close()

if __name__=='__main__':unittest.main()
