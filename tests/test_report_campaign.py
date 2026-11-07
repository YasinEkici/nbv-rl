"""Exercise every final report branch using registry-contained synthetic metrics."""
import csv
import json
from pathlib import Path


def test_complete_hypothesis_report_uses_only_synthetic_registry(tmp_path):
    from nbv import evaluation
    from nbv.phase1_report import ARMS, BASELINES, test_report
    records=[]
    for split in ('train','test'):
        for label in [*ARMS,*BASELINES]:
            baseline=label in BASELINES
            method=label.split('/')[0]
            phi=None if baseline else label.split('/')[1]
            for kind in (['baseline'] if baseline else ['tail','final']):
                value=1.15 if method=='sarsa' else (1.18 if method=='myopic' else 1.2)
                eps=[dict(part=p,variant=f'{p}_v{v}',seed=0,success=True,censored=False,
                          normalized_cycle_time=value,cycle_time=value*10,passes_to_95=2,
                          coverage_at_5=1.,coverage_at_10=1.,auc_coverage_time=.8,total_joint_motion=2.,
                          repair_passes=0,nsc_at_5=1.,nsc_at_10=1.,raw_at_5=.8,raw_at_10=.8)
                     for p in (1,2,3) for v in range(5)]
                extra=dict(tag='p002test' if split=='test' else 'p002train',n_candidates=48,
                           baseline=baseline,episode_metrics=eps,phi=phi,weight_kind=kind)
                records.append(evaluation.registry_row(f'{split}-{label}-{kind}',method,'synthetic',split,0,
                    evaluation.aggregate(eps),extra,'v4','v1','synthetic',status='canonical',
                    plan='001' if baseline and split=='train' else '002'))
    registry=tmp_path/'registry.csv'
    evaluation.append_registry(registry,records)
    predictions=tmp_path/'predictions.json'
    predictions.write_text(json.dumps({'predictions':{'H1':'rejected','H-forward':'rejected','H-shaping':'borderline'}}))
    curves=tmp_path/'curves.json'
    curves.write_text(json.dumps({f'sarsa/{phi}/11':{'steps':[100000,2000000],'mean':[1.2,1.2]}
                                 for phi in ('linear','none','convex','value_estimate')}))
    out=tmp_path/'report.md'
    incident=tmp_path/'incident.json'
    incident.write_text(json.dumps({'opening_commit':'4850144'}))
    resumed=tmp_path/'resumed.json'
    resumed.write_text(json.dumps({'commit':'synthetic-resume'}))
    result=test_report(str(registry),str(out),n_boot=20,predictions_path=str(predictions),curves_path=str(curves),
                       incident_path=str(incident),resume_path=str(resumed))
    assert not result['H1']['supported']
    assert result['H-forward']['supported']
    assert not result['H-shaping']['supported']
    text=out.read_text(encoding='utf-8')
    assert 'hierarchical bootstrap interval' in text and 'Appendix: final weights' in text
    assert 'Train−test raw seconds' in text
    assert 'NSC@5' in text and 'Raw coverage@10' in text
    assert 'Success without repair' in text
    assert 'Incident 1' in text and '4850144' in text and 'synthetic-resume' in text
    from nbv.phase1_report import train_report
    train=tmp_path/'train.md'
    train_report(str(registry),str(train))
    protocol=Path('docs/PROTOCOL.md').read_text(encoding='utf-8')
    limitations=protocol.split('## Model limitations (reported with every result)\n',1)[1].split('\n## ',1)[0].strip()
    assert limitations in text
    assert limitations in train.read_text(encoding='utf-8')
