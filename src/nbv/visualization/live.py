"""Local VAL-only inference/replay demo. Never writes experiments or opens the test holdout."""
import copy
import csv
import json
from pathlib import Path
import time

import numpy as np
from nbv import baselines,evaluation,parts,robot
from nbv.env import NBVEnv,evaluate_policy
from nbv.agents.frozen import FrozenLinearPolicy
from nbv.test_evaluation import checkpoint_matches,verify_artifact_hashes,checkpoint_identity
from nbv.visualization.replay import require_exact,ReplayMismatch,source_records,digest,SUMMARY_FIELDS,STEP_FIELDS
from nbv.visualization.demo import check_bundle,validate_document,decode_array

METHODS=('greedy_cost','mcts','sarsa/linear','sarsa/value_estimate','regq/linear','ilp_tsp','joint_opt')


def allowed_variant(part,index,manifest=None):
    manifest=manifest or parts.load_yaml('configs/split_v1.yaml')
    fid=int(part)
    allowed={p['file_id'] for p in manifest['parts'] if p['split']=='val'}
    test={p['file_id'] for p in manifest['parts'] if p['split']=='test'}
    if fid in test or fid not in allowed or type(index) is not int or index not in range(5):
        raise ValueError('VAL parts and variants v0-v4 only; closed test/train/unknown IDs rejected')
    return f'{fid}_v{index}'


def verify_live(actual,expected):
    fields=['sequence','cycle_time','success','normalized_cycle_time']
    fields+= [k for k in SUMMARY_FIELDS if k not in fields and k in expected]
    mismatches=[]
    for key in fields:
        try:require_exact(actual[key],expected[key],key)
        except (ReplayMismatch,KeyError,TypeError):mismatches.append(key)
    return dict(exact=not mismatches,mismatches=mismatches,matched_fields=[k for k in fields if k not in mismatches],
        equality='IEEE-754 binary64 bytes from full-precision JSON; no tolerance',
        final_coverage_status='derived live/replay coverage; consistent with stored partial oracle' if not mismatches else 'MISMATCH: do not claim recorded-result reproduction')


class SourceStore:
    """Read only canonical VAL sources, and verify checkpoint bytes before inference."""
    def __init__(self,weights_path=None):
        self.weights_path=Path(weights_path) if weights_path is not None else None
        self.manifest=parts.load_yaml('configs/split_v1.yaml')
        with Path('results/registry.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
        self.rows=[r for r in rows if r['status']=='canonical' and r['part_split']=='val' and r['env_version']=='v4']
        self.pretest=json.loads(Path('results/pretest_manifest_v1.json').read_text(encoding='utf-8'))
        self.selected=json.loads(Path('results/selected_settings_v1.json').read_text(encoding='utf-8'))
        if digest('results/selected_settings_v1.json')!=self.pretest['settings_sha256']:raise ValueError('frozen selected settings hash mismatch')
        self.replay=json.loads(Path('results/visualization/replay_validation_v1.json').read_text(encoding='utf-8'))
        if digest('results/registry.csv')!=self.replay['registry_sha256']:raise ValueError('frozen registry hash mismatch')
        self.records={};self.rng_states={}

    def row(self,method,seed):
        if method not in METHODS or seed not in range(10):raise ValueError('canonical method and seed 0-9 required')
        if method in ('greedy_cost','ilp_tsp','joint_opt') and seed!=0:raise ValueError('deterministic baseline uses seed 0 identity')
        if '/' in method:
            name,phi=method.split('/')
            found=[r for r in self.rows if r['method']==name and r['seed']==str(seed)
                and json.loads(r['extra_json']).get('tag')=='p002' and json.loads(r['extra_json']).get('phi')==phi]
        else:
            found=[r for r in self.rows if r['plan']=='001' and r['method']==method
                and (int(r['seed'] or 0)==seed) and json.loads(r['extra_json']).get('n_candidates')==48]
        if len(found)!=1:raise ValueError('unique canonical VAL source required')
        return found[0]

    def episodes(self,method,seed):
        row=self.row(method,seed);rid=row['run_id']
        if rid not in self.records:
            path=Path('outputs')/rid/'episodes.jsonl'
            expected=self.pretest['runs'][rid]['sha256']['episodes.jsonl'] if '/' in method else self.replay['sources'][rid]['sha256']
            if digest(path)!=expected:raise ValueError('canonical VAL episode hash mismatch')
            self.records[rid]=source_records(row,'outputs')
        return self.records[rid]

    def expected(self,method,seed,variant):
        fid,index=variant.split('_v');allowed_variant(int(fid),int(index),self.manifest)
        return next(r for r in self.episodes(method,seed) if r['variant']==variant)

    def reference(self,variant):return self.expected('ilp_tsp',0,variant)['cycle_time']

    def weights(self,method,seed):
        row=self.row(method,seed);rid=row['run_id'];entry=self.pretest['runs'].get(rid)
        if entry is None:raise ValueError('checkpoint absent from pretest manifest')
        directory=Path('outputs')/rid
        verify_artifact_hashes(directory,entry['sha256'])
        if checkpoint_identity(row,self.selected)!=entry['identity'] or not checkpoint_matches(directory,row,self.selected):
            raise ValueError('checkpoint identity mismatch')
        with np.load(directory/'weights.npz') as z:original=z['tail'].copy()
        if self.weights_path is None:return original
        from nbv.policy_weights import ARM_NAMES,load_policy
        published_info=json.loads(Path('provenance/publication.json').read_text(encoding='utf-8'))['weights']
        name,phi=method.split('/')
        published=load_policy(self.weights_path,ARM_NAMES[(name,phi)],seed,expected_sha256=published_info['sha256'])
        if published.tobytes()!=original.tobytes():raise ValueError('published tail weights differ from original checkpoint')
        return published

    def mcts_rng(self,seed,variant):
        rng=np.random.default_rng([seed,48,7]) # canonical METHODS.index('mcts') in run_baselines.py
        for record in self.episodes('mcts',seed):
            vid=record['variant']
            if vid==variant:return rng
            key=(seed,vid)
            if key not in self.rng_states:
                fid,index=vid.split('_v');allowed_variant(int(fid),int(index),self.manifest)
                env=NBVEnv([vid]);res=evaluate_policy(env,vid,lambda e:baselines.make_policy('mcts',e,rng))
                actual=summary(res,self.reference(vid))
                if not verify_live(actual,record)['exact']:raise ValueError(f'MCTS RNG preparation mismatch at {vid}')
                self.rng_states[key]=copy.deepcopy(rng.bit_generator.state)
            else:rng.bit_generator.state=copy.deepcopy(self.rng_states[key])
        raise ValueError('VAL variant absent from canonical MCTS order')


def summary(result,reference):
    return {'sequence':[s['pass'] for s in result['steps']], 'normalized_cycle_time':result['cycle_time']/reference,
            'censored':result['censored'],**evaluation.episode_metrics(result)}


class LiveSession:
    def __init__(self,store,part,index,method,seed):
        self.variant=allowed_variant(part,index,store.manifest);self.method=method;self.seed=seed
        self.expected=store.expected(method,seed,self.variant);self.reference=store.reference(self.variant)
        self.source_run_id=store.row(method,seed)['run_id']
        weights=store.weights(method,seed) if '/' in method else None
        rng=store.mcts_rng(seed,self.variant) if method=='mcts' else np.random.default_rng(0)
        self.env=NBVEnv([self.variant]);self.obs,_=self.env.reset(options={'variant':self.variant})
        self.recorded=method in ('ilp_tsp','joint_opt')
        self.policy=None if self.recorded else FrozenLinearPolicy(self.env.nominal,weights,self.env.k_max,self.env.target) if weights is not None else baselines.make_policy(method,self.env,rng)
        self.steps=[];self.joints=[];self.done=False;self.summary=None;self.verification=None;self.decision_ms=0.

    def step(self):
        if self.done:return None
        before=time.perf_counter();q_start=self.obs['q'].copy();index=len(self.steps)
        if self.recorded:action=self.expected['sequence'][index] if index<len(self.expected['sequence']) else None
        else:action=self.policy(self.obs,self.env.nominal,self.env.action_masks())
        if action is None:
            self.env.finish();self._finish();return None
        _,sweep=robot.transfer(q_start,self.env.nominal.q_plan[action],self.env.qlim,self.env.v,self.env.a)
        self.obs,_,terminated,truncated,info=self.env.step(int(action))
        flat=info['agent']|info['eval'];repair=bool(self.expected.get('repair',[False]*len(self.expected['sequence']))[index]) if self.recorded else False
        self.steps.append({k:flat[k] for k in STEP_FIELDS if k!='repair'}|{'repair':repair})
        self.joints.append(dict(q_start=q_start.tolist(),q_samples=sweep.tolist(),q_end=self.obs['q'].tolist(),elapsed_before_home=float(self.env._t_move+self.env._t_scan)))
        self.decision_ms=(time.perf_counter()-before)*1000
        if terminated or truncated or (self.recorded and len(self.steps)==len(self.expected['sequence'])):
            if not (terminated or truncated):self.env.finish()
            self._finish()
        s=self.steps[-1];j=self.joints[-1]
        return dict(kind='scan',pass_index=int(action),t_move_s=s['t_move'],t_scan_s=s['t_scan'],t_home_s=0.,
            cumulative_time_s=j['elapsed_before_home'],primary_coverage=s['coverage'],joint_start_rad=j['q_start'],
            joint_samples_rad=j['q_samples'],joint_end_rad=j['q_end'],repair=repair)

    def _finish(self):
        self.done=True
        if self.steps:self.steps[-1].update(t_home=self.env._t_home,cycle_time=self.env.cycle_time,total_joint_motion=self.env._joint_motion)
        success=bool(self.steps) and self.steps[-1]['coverage']>=self.env.target
        result=dict(steps=self.steps,cycle_time=self.env.cycle_time,success=success,censored=not success,
            passes_executed=len(self.steps),passes_to_target=len(self.steps) if success else None,
            repair_passes=sum(s['repair'] for s in self.steps),total_joint_motion=self.env._joint_motion)
        self.summary=summary(result,self.reference)
        self.verification=verify_live(self.summary,self.expected)


def recorded_session(part,method,directory='results/demo'):
    allowed_variant(part,0)
    check_bundle(directory) # no parsing/animation until hashes pass
    root=Path(directory)
    common=validate_document(json.loads((root/'common.json').read_text(encoding='utf-8')))
    data=validate_document(json.loads((root/f'part_{part}_v0.json').read_text(encoding='utf-8')))
    episode=next((e for e in data['episodes'] if e['method']==method),None)
    if episode is None:raise ValueError('method absent from recorded bundle')
    return episode|dict(mode='recorded bundle',part_data=data,common=common)


def create_server(port=8080,*,load_initial=True,recorded=False,weights_path=None):
    import viser
    from nbv.visualization.live_scene import DemoController
    from nbv.visualization.recorded_controller import RecordedController
    server=viser.ViserServer(host='127.0.0.1',port=port,label='NBV / VAL only')
    try:
        server._nbv_controller=RecordedController(server,load_initial=load_initial) if recorded else DemoController(server,load_initial=load_initial,weights_path=weights_path)
    except Exception:
        server.stop();raise
    return server


def validate_headless():
    """Persist a small, fixed VAL reproduction diagnostic, never a registry result."""
    state=evaluation.git_state()
    if state['dirty']:raise ValueError('commit live implementation before recording validation provenance')
    before=digest('results/registry.csv');bundle=check_bundle('results/demo');store=SourceStore();records=[]
    for method in METHODS:
        index=1 if method=='mcts' else 0
        session=LiveSession(store,271875,index,method,0)
        while not session.done:session.step()
        records.append(dict(method=method,part=271875,variant=session.variant,seed=0,source_run_id=session.source_run_id,
            mode='recorded CAD plan' if session.recorded else 'live frozen policy',
            verification=session.verification,actual=session.summary,
            expected={k:session.expected[k] for k in session.verification['matched_fields']}))
        if not session.verification['exact']:raise ValueError(f'headless VAL mismatch: {method}')
    if digest('results/registry.csv')!=before or check_bundle('results/demo')!=bundle:raise ValueError('protected input changed')
    report=dict(schema_version=1,commit=state['commit'],registry_sha256=before,bundle_manifest_sha256=bundle['manifest_sha256'],
        cases=records,all_exact=True,policy_training=False,test_access=False,registry_outputs_writes=False,
        description='Headless fixed-case VAL reproduction only, no selection; MCTS uses a later variant to exercise canonical RNG-prefix restoration.')
    target=Path('results/visualization/live_validation_v1.json')
    target.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    return report
