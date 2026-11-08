"""Style, ranking, paired arithmetic and test-holdout I/O boundaries."""
import csv
import json
from pathlib import Path
import numpy as np
import pytest


def test_shared_styles_are_okabe_ito_and_distinguish_more_than_eight_methods():
    from nbv.visualization.styles import PALETTE,STYLES
    assert PALETTE==('#E69F00','#56B4E9','#009E73','#F0E442','#0072B2','#D55E00','#CC79A7','#000000')
    assert len(STYLES)==13
    triples=[(s['color'],s['marker'],s['linestyle']) for s in STYLES.values()]
    assert len(set(triples))==13 and all(s['color'] in PALETTE for s in STYLES.values())
    assert STYLES['sarsa/linear']['color']=='#0072B2'
    assert STYLES['sarsa/value_estimate']['color']=='#E69F00'


def test_learning_highlights_use_official_score_lower_median_and_seed_ties():
    from nbv.visualization.figure_data import rank_runs
    runs=[{'seed':s,'score':(2 if s==0 else 1 if s==1 else s+3),'curve':[0 if s==0 else 100,s+1]} for s in range(10)]
    result=rank_runs(runs)
    assert result['best']==1 and result['median']==result['ordered'][4] and result['worst']==9
    assert result['alternative_curve_mean_order'][0]==0
    assert rank_runs([r|{'score':1.} for r in runs])['ordered']==list(range(10))
    assert 'lower median' in result['rule']


def test_test_plot_data_never_opens_cache_output_or_environment(tmp_path,monkeypatch):
    from nbv.visualization.figure_data import RegistryFigures
    import nbv.env
    fields=['run_id','status','part_split','method','extra_json','commit','seed']
    path=tmp_path/'registry.csv'
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for method in ('sarsa','myopic'):
            eps=[dict(part=p,variant=f'{p}_v0',seed=0,normalized_cycle_time=1.2 if method=='sarsa' else 1.23,
                      cycle_time=12.,success=True,total_joint_motion=10.) for p in (1,2,3)]
            writer.writerow(dict(run_id='p002test-NO-OUTPUT',status='canonical',part_split='test',method=method,
                extra_json=json.dumps(dict(tag='p002test',phi='linear',weight_kind='tail',episode_metrics=eps)),commit='synthetic',seed=0))
    monkeypatch.setattr(np,'load',lambda *a,**kw:pytest.fail('plotting read a cache'))
    monkeypatch.setattr(nbv.env,'NBVEnv',lambda *a,**kw:pytest.fail('plotting constructed environment'))
    original=Path.open
    def guarded(p,*a,**kw):
        if 'outputs' in p.parts or 'cache' in p.parts: pytest.fail('plotting opened protected output/cache')
        return original(p,*a,**kw)
    monkeypatch.setattr(Path,'open',guarded)
    data=RegistryFigures(path)
    assert len(data.test_records())==6 and data.paired()['mean']==pytest.approx(-.03)


def test_paired_figure_matches_frozen_verdict_at_full_and_display_precision():
    from nbv.visualization.figure_data import RegistryFigures
    value=RegistryFigures().paired()['mean']
    recorded=json.loads(Path('results/test_v1.json').read_text(encoding='utf-8'))['H-forward']['mean_difference']
    assert value.hex()==recorded.hex()
    assert format(value,'.15f')=='-0.019894820622543'


def test_gallery_rejects_real_test_id_before_reading_cache(monkeypatch):
    from nbv.visualization.assets import require_scene_part
    manifest={'parts':[{'file_id':1,'split':'val'},{'file_id':2,'split':'test'}]}
    assert require_scene_part(1,manifest)=='val'
    with pytest.raises(ValueError,match='closed'): require_scene_part(2,manifest)


def test_gallery_credit_is_visible_and_does_not_guess_license_version():
    from nbv.visualization.figures import gallery_credit
    source=dict(part=1,attribution=dict(name='Toy',author='Author',thing_url='https://www.thingiverse.com/thing:2',share_alike=True))
    credit=gallery_credit(source)
    assert all(word in credit for word in ('Author','CC BY-SA','version unknown','thing:2'))
