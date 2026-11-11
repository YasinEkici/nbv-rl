"""Inspect actual drawn artists, viewer traces and clean-worktree exports."""
import json
from pathlib import Path
import numpy as np
import pytest


def _synthetic_scene(part,**kwargs):
    """No CAD series live in gallery panels; isolate styling from cache access."""
    return dict(part=part,points=np.array([[0.,0.,0.],[1.,1.,1.],[2.,0.,1.]]),
        normals=np.tile([0.,0.,1.],(3,1)),nominal_measured=np.ones((48,3),bool),
        plan=dict(reachable=np.ones(48,bool),ik_reachable=np.ones(48,bool),
                  direction=np.tile([0.,0.,1.],(48,1))),
        attribution=dict(name='Synthetic surface',author='Fixture',share_alike=False,thing_url='https://example.com'))


CASES=[('report',stem,None) for stem in (
    'p004_a_quality_latency','p004_b_learning_runs','p004_c_test_distributions','p004_c2_paired_forward',
    'p004_d_val_coverage_time','p004_e_val_time_components','p004_e_test_motion_time',
    'p004_f_val_gallery','p004_f_train_gallery','p004_g_val_candidate_sphere','p004_g_train_candidate_sphere')]
CASES += [('slide',stem,stage) for stem in (
    'p004b_a_quality_latency','p004b_b_learning','p004b_c_test_parts','p004b_c2_paired_forward',
    'p004b_e_val_time_components','p004b_collision_rates') for stage in ('a','b')]


@pytest.mark.parametrize('family,stem,stage',CASES)
def test_every_drawn_cad_marker_is_hollow_after_canvas_draw(monkeypatch,family,stem,stage):
    from matplotlib.collections import PathCollection
    from matplotlib.markers import MarkerStyle
    from matplotlib.colors import to_rgba
    import matplotlib.pyplot as plt
    from nbv.visualization import design_figures as renderer,theme
    monkeypatch.setattr(renderer.assets,'load_scene',_synthetic_scene)
    replay=[dict(method=m,variant='271875_v0',seed=0,joint_trace=[{'elapsed_before_home':1.}],
                 steps=[{'coverage':.95}],summary={'cycle_time':2.}) for m in ('ilp_tsp','joint_opt')]
    monkeypatch.setattr(renderer,'validated_records',lambda:(replay,None))
    folder=Path('results/figures')/('slides' if family=='slide' else '')
    old=json.loads((folder/(stem+'.json')).read_text(encoding='utf-8'))
    kind=(renderer.SLIDES if family=='slide' else renderer.REPORTS)[stem]
    fig,_,_,_=renderer.draw(kind,old,'tr' if family=='slide' else 'en',stage)
    try:
        fig.canvas.draw();found=set()
        for ax in fig.axes:
            for line in ax.lines:
                for method in ('ilp_tsp','joint_opt'):
                    if line.get_marker()==theme.base_styles()[method]['marker']:
                        found.add(method)
                        assert to_rgba(line.get_markerfacecolor())[3]==0,f'{stem}: {method} line marker is filled'
            for artist in ax.collections:
                if not isinstance(artist,PathCollection):continue
                for method in ('ilp_tsp','joint_opt'):
                    marker=MarkerStyle(theme.base_styles()[method]['marker'])
                    expected=marker.get_path().transformed(marker.get_transform()).vertices
                    if any(np.array_equal(path.vertices,expected) for path in artist.get_paths()):
                        found.add(method);faces=artist.get_facecolors()
                        assert len(faces)==0 or np.all(faces[:,3]==0),f'{stem}: {method} marker is filled'
        required=({'ilp_tsp'} if family=='slide' else {'ilp_tsp','joint_opt'}) if kind=='c' else (
            {'ilp_tsp','joint_opt'} if kind in ('motion','collision') else set())
        assert required<=found,'Expected CAD artists were not inspected'
    finally:plt.close(fig)


def test_corrected_motion_export_preserves_every_field_except_render_commit():
    import hashlib
    root=Path('results/figures');stem='p004_e_test_motion_time'
    old=json.loads((root/(stem+'_v2.json')).read_text(encoding='utf-8'))
    current=json.loads((root/(stem+'_v3.json')).read_text(encoding='utf-8'))
    assert current['render_commit']!=old['render_commit']
    current['render_commit']=old['render_commit']
    assert json.dumps(current,sort_keys=True,ensure_ascii=False)==json.dumps(old,sort_keys=True,ensure_ascii=False)
    manifest=json.loads((root/'manifest_v3.json').read_text(encoding='utf-8'))
    for name,sha in manifest['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha
