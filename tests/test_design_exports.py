"""Versioned figures preserve source payloads and inspect actual rendered text."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest


def test_numeric_and_source_payloads_are_exact_and_unknown_fields_rejected():
    from nbv.visualization.design_export import validate_invariance
    old={'commit':'old','title':'Old','values':[1.,-.019894820622543344],
         'source_run_ids':['run'],'source_commits':{'run':'abc'},'rankings':{'best':0}}
    new=copy.deepcopy(old)|{'title':'New','theme_version':'v2','render_commit':'new','base_style':{}}
    validate_invariance(old,new)
    bad=copy.deepcopy(new);bad['values'][1]=float(np.nextafter(bad['values'][1],np.inf))
    with pytest.raises(ValueError,match='payload'):validate_invariance(old,bad)
    with pytest.raises(ValueError,match='field'):validate_invariance(old,new|{'undeclared':True})
    bad=copy.deepcopy(new);bad['source_commits']['run']='different'
    with pytest.raises(ValueError,match='payload'):validate_invariance(old,bad)


def test_actual_drawn_text_rejects_internal_ids_and_decimal_points():
    import matplotlib.pyplot as plt
    from nbv.visualization.design_export import setup_fonts,rendered_text
    setup_fonts(slide=True)
    fig,ax=plt.subplots();ax.set_title('greedy_cost');ax.plot([0,1],[0,1])
    ax.set_xticks([]);ax.set_yticks([])  # Isolate each deliberately invalid title from numeric tick labels.
    with pytest.raises(ValueError,match='identifier'):rendered_text(fig,'tr')
    ax.set_title('0.02')
    with pytest.raises(ValueError,match='decimal'):rendered_text(fig,'tr')
    plt.close(fig)


def test_font_glyphs_and_portable_svg_and_embedded_pdf(tmp_path):
    import matplotlib.pyplot as plt
    from nbv.visualization.design_export import setup_fonts,rendered_text,save_figure
    from PIL import Image
    setup_fonts(slide=True)
    fig,ax=plt.subplots(figsize=(12.8,7.2),dpi=150)
    ax.text(.2,.5,'İıŞşĞğÜüÖöÇç Φ λ − · 0,02');ax.set_axis_off()
    text=rendered_text(fig,'tr');assert any('Φ' in s for s in text)
    save_figure(fig,tmp_path/'example',{'commit':'source','values':[1.]},language='tr',slide=True)
    with Image.open(tmp_path/'example.png') as im:assert im.size==(1920,1080)
    assert '<text' not in (tmp_path/'example.svg').read_text(encoding='utf-8')
    assert (tmp_path/'example.pdf').stat().st_size>0
    metadata=json.loads((tmp_path/'example.json').read_text(encoding='utf-8'))
    assert metadata['rendered_text']==text and metadata['font']['family']=='IBM Plex Sans'


def test_global_display_order_and_main_trio_adjacency_are_reported():
    from nbv.visualization.design_export import display_summary
    from nbv.visualization.theme import TRIO
    d=display_summary();values=[d['scores'][m] for m in d['order'] if m in d['scores']]
    assert values==sorted(values)
    indices=[d['order'].index(m) for m in TRIO]
    assert d['trio_adjacent']==(max(indices)-min(indices)==2)
