"""Global method identities and reproducible, accessibility-based palette selection."""
import json
from pathlib import Path
import numpy as np

CORE=('greedy_cost','mcts','ilp_tsp','joint_opt','sarsa/linear','sarsa/value_estimate','regq/linear')


def test_method_identity_does_not_depend_on_figure_subset():
    from nbv.visualization.styles import styles_for
    a=styles_for(['sarsa/linear','regq/linear','sarsa/value_estimate','greedy_cost','mcts'])
    b=styles_for(CORE)
    assert a['mcts']==b['mcts']


def test_maximin_has_known_optimum_and_lexicographic_tie_rule():
    from nbv.visualization.palette import select_maximin
    d=np.array([[0,1,5,5,2],[1,0,4,4,2],[5,4,0,5,1],[5,4,5,0,1],[2,2,1,1,0]],float)
    assert select_maximin(d,3)==(0,2,3)
    assert select_maximin(np.ones((5,5))-np.eye(5),3)==(0,1,2)


def test_committed_theme_has_unique_core_colors_and_family_codes(monkeypatch):
    from nbv.visualization import theme
    import sys
    monkeypatch.setitem(sys.modules,'colorspacious',None)
    t=theme.load_theme();styles=theme.base_styles()
    assert t['version']=='v2' and len(styles)==13
    assert len({(s['color'],s['marker'],s['linestyle']) for s in styles.values()})==13
    assert len({styles[n]['color'] for n in CORE})==7
    for name,s in styles.items():
        assert s['linestyle']==('-' if '/' in name else '--' if name in ('ilp_tsp','joint_opt') else ':')
        assert s['fillstyle']==('none' if name in ('ilp_tsp','joint_opt') else 'full')
        assert theme.method_name(name,'tr') and theme.method_name(name,'en')


def test_palette_report_contains_exhaustive_selection_and_valid_runtime_artifact():
    from nbv.visualization.palette import select_maximin
    t=json.loads(Path('results/visualization/theme_v2.json').read_text(encoding='utf-8'))
    r=json.loads(Path('results/visualization/palette_check_v1.json').read_text(encoding='utf-8'))
    distances=np.array(r['minimum_distances'])
    assert r['subset_count']==120
    chosen=tuple(r['selected_seven']);assert chosen==select_maximin(distances,7)
    triple=select_maximin(distances[np.ix_(chosen,chosen)],3)
    assert tuple(r['selected_triple'])==tuple(chosen[i] for i in triple)
    assert r['selected_seven']==t['palette']['selected_seven']
    assert set(r['vision_cases'])=={'normal','deuteranomaly','protanomaly','tritanomaly'}
    assert all(np.isfinite(v).all() for v in r['vision_cases'].values())


def test_rendering_sources_have_no_local_color_or_display_name_literals():
    import ast,re
    from nbv.visualization.theme import DEFINITIONS
    visible={label for method,labels in DEFINITIONS.items() for label in labels[1:] if label!=method}
    failures=[]
    for p in [*Path('src/nbv/visualization').glob('*.py'),*Path('scripts').glob('*.py')]:
        if p.name=='theme.py':continue
        tree=ast.parse(p.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node,ast.Constant) and isinstance(node.value,str):
                if re.search(r'#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?\b',node.value) or node.value.startswith('tab:') or node.value in visible:
                    failures.append(f'{p}:{node.lineno}:{node.value[:70]}')
            if isinstance(node,ast.keyword) and node.arg and 'color' in node.arg:
                if isinstance(node.value,(ast.Tuple,ast.List)) and all(isinstance(v,ast.Constant) for v in node.value.elts):
                    failures.append(f'{p}:{node.lineno}:literal color tuple')
    assert not failures,'\n'.join(failures)
