"""Every continuation precondition is exercised on synthetic files, before held-out use."""
import json
from pathlib import Path
import pytest


@pytest.fixture
def recovery(tmp_path, monkeypatch):
    from nbv import incident_resume as m, test_evaluation as gate
    from nbv.test_inputs import sha256
    outputs=tmp_path/'outputs'; outputs.mkdir()
    pretest=tmp_path/'pretest.json'; pretest.write_text('{}')
    opening=outputs/'p002-test-opened.json'
    opening.write_text(json.dumps({'commit':'4850144','pretest_manifest_sha256':sha256(pretest),'official_runs':['source']}))
    incident=outputs/'p002-test-incident-1.json'
    incident.write_text(json.dumps(dict(incident=1,opening_commit='4850144',opening_sha256=sha256(opening),
        performance_information_produced=False,evidence=dict(registry_test_rows=0,completion_marker_exists=False,test_episode_files=[]))))
    manifest=tmp_path/'inputs.json'
    manifest.write_text(json.dumps({'version':1,'inputs':[{'path':str(incident),'sha256':sha256(incident)},
                                                       {'path':str(opening),'sha256':sha256(opening)}]}))
    monkeypatch.setattr(gate,'preflight',lambda **kw:([{'run_id':'source'}],dict(commit='new',dirty=False,pretest_manifest_sha256=sha256(pretest))))
    monkeypatch.setattr(gate,'load_rows',lambda path:[])
    monkeypatch.setattr(gate,'execute',lambda *a,**kw:{'synthetic_complete':True})
    monkeypatch.setattr(m.evaluation,'_git',lambda *a:'tracked')
    monkeypatch.setattr(m.evaluation,'git_state',lambda:dict(commit='new',dirty=False))
    monkeypatch.setattr(m,'passing_suite',lambda:{'returncode':0,'output':'synthetic passing suite'})
    monkeypatch.setattr(m,'dry_run',lambda **kw:{'missing':[],'inputs':[{'path':str(incident)},{'path':str(opening)}]})
    return m,outputs,pretest,manifest


@pytest.mark.parametrize('condition',[
    'missing_opening','changed_opening','missing_incident','test_row','completed','episodes',
    'dirty','untracked_pretest','untracked_inputs','changed_pretest','changed_inputs','missing_input','pytest_failed','wrong_incident','repeat'])
def test_resume_rejects_every_unsafe_condition_before_execution(recovery, monkeypatch, condition):
    m,outputs,pretest,manifest=recovery
    from nbv import test_evaluation as gate
    kwargs=dict(incident=1,outputs=outputs,pretest_path=pretest,input_path=manifest)
    if condition=='missing_opening': (outputs/'p002-test-opened.json').unlink()
    if condition=='changed_opening': (outputs/'p002-test-opened.json').write_text('changed')
    if condition=='missing_incident': (outputs/'p002-test-incident-1.json').unlink()
    if condition=='test_row': monkeypatch.setattr(gate,'load_rows',lambda path:[{'part_split':'test'}])
    if condition=='completed': (outputs/'p002-test-completed.json').write_text('{}')
    if condition=='episodes':
        d=outputs/'p002test-synthetic'; d.mkdir(); (d/'episodes.jsonl').write_text('')
    if condition=='dirty': monkeypatch.setattr(gate,'preflight',lambda **kw:([{'run_id':'source'}],{'dirty':True}))
    if condition.startswith('untracked'):
        target=pretest if condition=='untracked_pretest' else manifest
        def git(*args):
            if str(target) in args: raise ValueError('not committed')
            return 'tracked'
        monkeypatch.setattr(m.evaluation,'_git',git)
    if condition=='changed_pretest': pretest.write_text('changed')
    if condition=='changed_inputs':
        data=json.loads(manifest.read_text()); data['inputs'][0]['sha256']='wrong'; manifest.write_text(json.dumps(data))
    if condition=='missing_input': monkeypatch.setattr(m,'dry_run',lambda **kw:{'inputs':[],'missing':['missing']})
    if condition=='pytest_failed': monkeypatch.setattr(m,'passing_suite',lambda:{'returncode':1,'output':'synthetic failure'})
    if condition=='wrong_incident': kwargs['incident']=2
    if condition=='repeat': (outputs/'p002-test-resumed-1.json').write_text('{}')
    monkeypatch.setattr(gate,'execute',lambda *a,**kw:pytest.fail('evaluation started despite failed guard'))
    with pytest.raises((ValueError,FileNotFoundError,FileExistsError)):
        m.resume_after_incident(**kwargs)


def test_exclusive_resume_is_single_use_and_preserves_original_claim(recovery):
    m,outputs,pretest,manifest=recovery
    original=(outputs/'p002-test-opened.json').read_bytes()
    kwargs=dict(incident=1,outputs=outputs,pretest_path=pretest,input_path=manifest)
    assert m.resume_after_incident(**kwargs)=={'synthetic_complete':True}
    assert (outputs/'p002-test-resumed-1.json').exists()
    assert (outputs/'p002-test-opened.json').read_bytes()==original
    with pytest.raises(FileExistsError): m.resume_after_incident(**kwargs)


def test_baseline_uses_frozen_ilp_plan_instead_of_resolving(tmp_path, monkeypatch):
    import importlib.util
    from nbv import env, baselines
    from test_env import synthetic
    original=env.NBVEnv
    monkeypatch.setattr(env,'NBVEnv',lambda variants,**kw:original(variants,loader=lambda vid:synthetic(vid),**kw))
    spec=importlib.util.spec_from_file_location('entry',Path('scripts/run_baselines.py'))
    entry=importlib.util.module_from_spec(spec); spec.loader.exec_module(entry)
    monkeypatch.setattr(entry,'d_plan95',lambda *a:{1:2})
    monkeypatch.setattr(baselines,'ilp_tsp_plan',lambda *a,**kw:pytest.fail('re-solved frozen plan'))
    frozen=tmp_path/'ilp.json'; frozen.write_text(json.dumps({'1':{'plan':[0,1],'info':{'ilp_status':'FROZEN'}}}))
    result=entry.run_task(dict(method='ilp_tsp',n=48,split='test',seed=0,variants=['1_v0'],run_id='synthetic',
                               d_plan='unused',ilp_plans=str(frozen)))
    assert result['episodes'][0]['planner']['ilp_status']=='FROZEN'
