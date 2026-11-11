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


@pytest.mark.parametrize('chosen',[
    'greedy_cost','mcts','ilp_tsp','joint_opt','sarsa/linear','sarsa/value_estimate','regq/linear'])
def test_viewer_selected_and_context_traces_match_global_line_styles(chosen):
    from types import SimpleNamespace
    from nbv.visualization.presentation import CoverageChart
    from nbv.visualization import theme
    class GUI:
        def add_plotly(self,figure,**kwargs):return SimpleNamespace(figure=figure)
    chart=CoverageChart(GUI())
    episodes=[dict(method=m,steps=[dict(cumulative_time_s=1.,primary_coverage=.95)]) for m in theme.CORE]
    chart.reset(episodes,chosen)
    traces={t.name:t for t in chart.handle.figure.data}
    expected={'-':'solid',':':'dot','--':'dash','-.':'dashdot'}
    for method in theme.CORE:
        trace=traces[theme.method_name(method,'tr')]
        assert trace.line.dash==expected[theme.base_styles()[method]['linestyle']],method
        assert tuple(trace.x)==(0.,1.) and tuple(trace.y)==(0.,.95)


def test_cache_free_export_starts_without_scratch_in_tracked_worktree():
    import os
    import subprocess
    import sys
    import tempfile
    root=Path(__file__).resolve().parents[1];scratch=root/'scratch'
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='004-final-check-',dir=scratch) as directory:
        container=Path(directory).resolve();assert container.is_relative_to(scratch.resolve())
        tree=container/'tree';assert tree.resolve().is_relative_to(scratch.resolve())
        subprocess.run(['git','worktree','add','--detach',str(tree),'HEAD'],cwd=root,check=True,capture_output=True)
        try:
            tracked=set(subprocess.check_output(['git','ls-files'],cwd=tree,text=True,encoding='utf-8').splitlines())
            actual={p.relative_to(tree).as_posix() for p in tree.rglob('*') if p.is_file()}
            assert actual==tracked|{'.git'}
            assert not any((tree/p).exists() for p in ('scratch','data/cache','outputs'))
            # Pending renderer edits run in memory; every file/resource remains the clean tracked HEAD.
            # Once committed, these overlays are byte-identical to the worktree source too.
            overlay={name:dict(path=path,source=(root/path).read_text(encoding='utf-8')) for name,path in {
                'nbv.visualization':'src/nbv/visualization/__init__.py',
                'nbv.visualization.design_figures':'src/nbv/visualization/design_figures.py'}.items()}
            program=r'''
import importlib.abc,importlib.util,json,os,sys
from pathlib import Path
payload=json.load(sys.stdin);tree=Path.cwd().resolve()
os.environ.pop('MPLCONFIGDIR',None)
class Overlay(importlib.abc.MetaPathFinder,importlib.abc.Loader):
 def find_spec(self,fullname,path=None,target=None):
  if fullname not in payload['modules']:return None
  filename=tree/payload['modules'][fullname]['path']
  return importlib.util.spec_from_file_location(fullname,filename,loader=self,
   submodule_search_locations=[str(filename.parent)] if filename.name=='__init__.py' else None)
 def create_module(self,spec):return None
 def exec_module(self,module):exec(compile(payload['modules'][module.__name__]['source'],module.__file__,'exec'),module.__dict__)
sys.meta_path.insert(0,Overlay())
import nbv
assert Path(nbv.__file__).resolve().is_relative_to(tree)
sys.argv=['scripts/report_figures.py','--figure','p004_a_quality_latency','--figure','p004_e_test_motion_time',
 '--output-dir','scratch/generated','--version','v3']
exec(compile(payload['entry'],str(tree/'scripts/report_figures.py'),'exec'),{'__name__':'__main__','__file__':str(tree/'scripts/report_figures.py')})
import matplotlib
assert Path(matplotlib.get_configdir()).resolve().is_relative_to(tree/'scratch')
assert Path(matplotlib.get_configdir()).is_dir()
'''
            env=os.environ|{'PYTHONPATH':str(tree/'src'),'MPLBACKEND':'Agg','PYTHONDONTWRITEBYTECODE':'1'}
            payload={'modules':overlay,'entry':(root/'scripts/report_figures.py').read_text(encoding='utf-8')}
            result=subprocess.run([sys.executable,'-c',program],input=json.dumps(payload),cwd=tree,
                                  env=env,text=True,encoding='utf-8',capture_output=True,timeout=90)
            assert result.returncode==0,result.stdout+'\n'+result.stderr
            manifest=json.loads((tree/'scratch/generated/manifest_v3.json').read_text(encoding='utf-8'))
            assert len(manifest['figures'])==2 and len(manifest['files'])==8
            assert not (tree/'data/cache').exists() and not (tree/'outputs').exists()
            assert not subprocess.check_output(['git','status','--porcelain'],cwd=tree,text=True).strip()
        finally:
            assert tree.resolve().is_relative_to(scratch.resolve())
            subprocess.run(['git','worktree','remove','--force',str(tree)],cwd=root,check=True,capture_output=True)
