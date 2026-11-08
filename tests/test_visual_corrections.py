"""Presentation-only invariants; no scientific values or test geometry change."""
import numpy as np
import pytest


@pytest.mark.parametrize('names',[
    ['sarsa/linear','regq/linear','sarsa/value_estimate','greedy_cost','mcts'],
    ['greedy_cost','mcts','ilp_tsp','joint_opt','sarsa/linear','sarsa/value_estimate','regq/linear'],
])
def test_small_figures_have_unique_okabe_ito_colors_and_no_duplicate_triples(names):
    from nbv.visualization.styles import styles_for,validate_styles,PALETTE
    styles=styles_for(names);validate_styles(styles)
    assert len({s['color'] for s in styles.values()})==len(names)
    assert all(s['color'] in PALETTE for s in styles.values())
    assert styles_for(list(reversed(names)))==styles
    corrupted={name:dict(style) for name,style in styles.items()}
    corrupted[names[1]]=dict(corrupted[names[0]])
    with pytest.raises(ValueError,match='duplicate'):validate_styles(corrupted)


def test_large_figures_encode_families_with_line_and_fill():
    from nbv.visualization.styles import STYLES,styles_for
    styles=styles_for(list(STYLES))
    for name,s in styles.items():
        if '/' in name: assert s['linestyle']=='-' and s['fillstyle']=='full'
        elif name in ('ilp_tsp','joint_opt'):assert s['linestyle']=='--' and s['fillstyle']=='none'
        else:assert s['linestyle']==':'


def test_paired_means_are_lines_and_keep_exact_values():
    import matplotlib.pyplot as plt
    from nbv.visualization.figures import draw_paired_reference
    split=-0.019894820622543344;part=-.03
    fig,ax=plt.subplots()
    artists=draw_paired_reference(ax,split,part)
    assert list(artists['split'].get_ydata())==[split,split]
    assert artists['part'].get_segments()[0][:,1].tolist()==[part,part]
    assert artists['split'].get_marker() in ('None',None,'')
    plt.close(fig)


def test_candidate_map_uses_azimuth_elevation_and_gallery_uses_actual_range():
    from nbv.visualization.figures import candidate_angles,gallery_range
    az,el=candidate_angles(np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]))
    assert np.array_equal(az,[0.,90.,0.]) and np.array_equal(el,[0.,0.,90.])
    scene={'nominal_measured':np.array([[1,0,1],[1,0,0],[1,1,1]],bool),'plan':{'reachable':np.array([1,1,0],bool)}}
    assert gallery_range([scene])==(1,2)
