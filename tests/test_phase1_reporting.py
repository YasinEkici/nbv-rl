"""Frozen evaluation completeness and pretest curve diagnosis, synthetic data only."""
import pytest


def test_curve_checkpoints_and_auc_use_all_registered_points():
    from nbv.phase1_report import curve_diagnosis
    curves={f'sarsa/{p}/11':dict(steps=[200000,1000000,2000000],mean=[v]*3,low=[v-.01]*3,high=[v+.01]*3)
            for p,v in [('linear',1.16),('none',1.18),('convex',1.17),('value_estimate',1.15)]}
    curves['regq/linear/11']=dict(steps=[200000,1000000,2000000],mean=[1.1]*3,low=[1.]*3,high=[1.2]*3)
    curves['sarsa/linear/11']['mean']=[1.16,1.16,1.20]
    result=curve_diagnosis(curves,1.162)
    assert result['dominant']=='oscillation'
    assert result['checkpoints']['sarsa/linear/11'][200000]['mean']==1.16
    assert result['auc']['linear']==pytest.approx(1.16*800000+1.18*1000000)
    assert result['auc_relative_range']==pytest.approx((1.18-1.15)/1.15)


def test_completed_test_campaign_requires_every_predeclared_key():
    from nbv.frozen_evaluation import campaign_complete
    from nbv.test_evaluation import ARMS
    baseline_seeds={m:(range(10) if m in ('random','mcts') else [0])
                    for m in ('random','fixed','greedy_ig','greedy_cost','ilp_tsp','mcts','joint_opt')}
    keys=[(m,None,'baseline',s) for m,seeds in baseline_seeds.items() for s in seeds]
    keys += [(m,p,k,s) for m,p in ARMS for k in ('tail','final') for s in range(10)]
    assert len(keys)==145
    assert campaign_complete(keys)==145
    with pytest.raises(ValueError): campaign_complete(keys[:-1])
    with pytest.raises(ValueError): campaign_complete(keys+[keys[0]])


def test_frozen_registry_record_retains_full_precision_episode_metrics(tmp_path):
    from nbv.frozen_evaluation import save_evaluation
    import csv,json
    e=dict(part=1,variant='1_v0',success=True,censored=False,normalized_cycle_time=1.123456789,
           cycle_time=12.3456789,passes_to_95=2,coverage_at_5=1.,coverage_at_10=1.,auc_coverage_time=.9,
           total_joint_motion=2.,repair_passes=0,nsc_at_5=1.,nsc_at_10=1.,raw_at_5=.8,raw_at_10=.8)
    row=save_evaluation('sarsa',0,[e],dict(source_run='source'),{'phi':'linear','weight_kind':'tail'},
                        'train',{'commit':'synthetic','dirty':False,'config_files':{}},
                        str(tmp_path/'registry.csv'),str(tmp_path/'outputs'),'p002train',2000000)
    extra=json.loads(row['extra_json'])
    assert extra['episode_metrics'][0]['normalized_cycle_time']==1.123456789
    assert extra['source_run']=='source'
    assert row['status']=='canonical' and row['part_split']=='train'
