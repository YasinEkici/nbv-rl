"""Exact full/partial replay oracles on synthetic data; no policy selection."""
import copy
import numpy as np
import pytest


def record(repair=False):
    from test_env import make_env
    from nbv.env import evaluate_sequence
    from nbv.evaluation import episode_metrics
    e=make_env()
    if repair:
        remaining=iter([3])
        result=evaluate_sequence(e,'1_v0',[0,1],lambda *args:next(remaining,None))
    else: result=evaluate_sequence(e,'1_v0',[0,1,3])
    return dict(variant='1_v0',part=1,sequence=[s['pass'] for s in result['steps']],
                steps=result['steps'],repair=[s['repair'] for s in result['steps']],censored=result['censored'],
                normalized_cycle_time=result['cycle_time']/10,**episode_metrics(result))


@pytest.mark.parametrize('damage',['sequence','one_ulp'])
def test_full_oracle_blocks_corrupted_sequence_or_step(damage):
    from nbv.visualization.replay import replay_episode,ReplayMismatch
    from test_env import make_env
    stored=record()
    if damage=='sequence': stored['sequence'][0],stored['sequence'][1]=stored['sequence'][1],stored['sequence'][0]
    else: stored['steps'][0]['t_move']=float(np.nextafter(stored['steps'][0]['t_move'],np.inf))
    with pytest.raises(ReplayMismatch): replay_episode(make_env(),stored,10.,stage='A')


@pytest.mark.parametrize('repair',[False,True])
def test_replay_exact_full_oracle_and_recorded_repairs_without_policy(monkeypatch,repair):
    from nbv.visualization.replay import replay_episode
    from nbv import baselines
    from nbv.agents.linear import LinearAgent
    from test_env import make_env
    monkeypatch.setattr(baselines,'make_policy',lambda *a:pytest.fail('policy constructor called'))
    monkeypatch.setattr(LinearAgent,'select',lambda *a:pytest.fail('agent action selection called'))
    original=record(repair)
    fresh=replay_episode(make_env(),original,10.,stage='A')
    assert fresh['verification']['exact'] and fresh['verification']['stage']=='A'
    assert [s['repair'] for s in fresh['steps']]==original['repair']
    assert len(fresh['joint_trace'])==len(original['sequence'])
    assert all(np.array_equal(t['q_end'],t['q_samples'][-1]) for t in fresh['joint_trace'])


def test_partial_oracle_checks_every_summary_and_labels_derived_final():
    from nbv.visualization.replay import replay_episode,ReplayMismatch,SUMMARY_FIELDS
    from test_env import make_env
    stored=record(); stored.pop('steps'); stored.pop('repair')
    fresh=replay_episode(make_env(),stored,10.,stage='B')
    assert set(SUMMARY_FIELDS)<=set(fresh['verification']['matched_fields'])
    assert fresh['verification']['final_coverage_status']=='derived from replay; consistent with partial oracle'
    bad=copy.deepcopy(stored); bad['auc_coverage_time']=float(np.nextafter(bad['auc_coverage_time'],np.inf))
    with pytest.raises(ReplayMismatch,match='auc_coverage_time'): replay_episode(make_env(),bad,10.,stage='B')


def test_replay_holdout_guard_rejects_before_read_or_construction(monkeypatch):
    from nbv.visualization.replay import require_allowed
    manifest={'variants':[{'variant_id':'1_v0','split':'val'},{'variant_id':'2_v0','split':'test'}]}
    assert require_allowed('1_v0',manifest)=='val'
    with pytest.raises(ValueError,match='closed'): require_allowed('2_v0',manifest)
