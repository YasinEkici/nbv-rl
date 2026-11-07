"""Full evaluation orchestration on generated 8-pass/60-sample fixtures, no real cache reads."""
import json
from pathlib import Path
import numpy as np
import pytest


@pytest.mark.parametrize('prepared',[False,True])
def test_full_test_campaign_is_complete_on_synthetic_parts(tmp_path,monkeypatch,prepared):
    from nbv import test_evaluation as module, env as envmod, parts, evaluation, baselines
    from test_env import synthetic
    from test_test_gate import synthetic_rows
    variants=[f'{fid}_v{v}' for fid in (1,2,3) for v in range(5)]
    fixture={vid:synthetic(vid,seed=int(vid.split('_')[0]),real_seed=20+int(vid[-1])) for vid in variants}
    original_env=envmod.NBVEnv
    monkeypatch.setattr(envmod,'NBVEnv',lambda ids,**kw:original_env(ids,loader=lambda vid:fixture[vid],**kw))
    original_yaml=parts.load_yaml
    monkeypatch.setattr(parts,'load_yaml',lambda path:({'variants':[{'variant_id':v,'split':'test'} for v in variants]}
                                                       if Path(path).name=='split_v1.yaml' else original_yaml(path)))
    monkeypatch.setattr(evaluation,'selected_cache_manifest',lambda paths:{'files':[],'sha256':'synthetic-only'})
    original_policy=baselines.make_policy
    monkeypatch.setattr(baselines,'make_policy',lambda method,e,rng:(baselines.greedy_cost_policy(e.v,e.a)
                                                                  if method=='mcts' else original_policy(method,e,rng)))
    original_spec=module.importlib.util.spec_from_file_location
    def spec(name,path,*args,**kw):
        result=original_spec(name,path,*args,**kw)
        if name=='baseline_entry':
            loader=result.loader
            class Loader:
                def create_module(self,s): return loader.create_module(s)
                def exec_module(self,m):
                    loader.exec_module(m)
                    m.d_plan95=lambda path,n:{1:3,2:3,3:3}
            result.loader=Loader()
        return result
    monkeypatch.setattr(module.importlib.util,'spec_from_file_location',spec)
    outputs=tmp_path/'outputs'; outputs.mkdir()
    (outputs/'p002-test-opened.json').write_text('{"synthetic":true}')
    selected=synthetic_rows()
    for r in selected:
        d=outputs/r['run_id']
        evaluation.write_run(d,{},dict(commit='synthetic',dirty=False),[],{})
        np.savez(d/'weights.npz',tail=np.zeros(11),final=np.zeros(11))
        (d/'training.json').write_text('{}')
    registry=tmp_path/'registry.csv'
    state=dict(commit='synthetic',dirty=False)
    if prepared:
        plan_dir=tmp_path/'plans'; plan_dir.mkdir()
        for fid in (1,2,3):
            joint,info=module.prepare_joint(fixture[f'{fid}_v0'].nominal)
            (plan_dir/f'test_{fid}_n48.json').write_text(json.dumps(dict(joint_plan=joint,joint_info=info,
                ilp_plan=info['ilp_hint'],ilp_info=info['ilp_info'])))
        monkeypatch.setattr(module,'prepare_joint',lambda *a:pytest.fail('re-solved frozen plan'))
        state['prepared_plan_dir']=str(plan_dir)
    result=module.execute(selected,state,str(registry),str(outputs))
    assert result['run_count']==145 and len(set(result['evaluation_runs']))==145
    written=module.load_rows(registry)
    assert len(written)==145
    assert all(r['commit']=='synthetic' and r['status']=='canonical' for r in written)
    assert all(len(json.loads(r['extra_json'])['episode_metrics'])==15 for r in written)
    assert (outputs/'p002-test-completed.json').exists()
