"""Offline exhaustive palette measurement. No colorspacious import in renderers."""
import itertools
import importlib.metadata
import json
from pathlib import Path
import platform
import numpy as np
from nbv.visualization import theme

def select_maximin(distances,k):
    distances=np.asarray(distances,float);best=None;best_score=-float('inf')
    for subset in itertools.combinations(range(len(distances)),k):
        score=min(distances[i,j] for i,j in itertools.combinations(subset,2))
        if score>best_score:best,best_score=subset,float(score)
    return best

def luminance(rgb):
    rgb=np.asarray(rgb,float)
    linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
    return linear@np.array([.2126,.7152,.0722])

def generate(directory='results/visualization'):
    from matplotlib import color_sequences
    from matplotlib.colors import to_rgb,to_hex
    from colorspacious import cspace_convert
    from nbv.visualization.figure_data import RegistryFigures
    from nbv.evaluation import git_state
    rgb=np.array([to_rgb(c) for c in color_sequences['petroff10']]);colors=[to_hex(c) for c in rgb]
    cases={};clipped={};raw={}
    for name in ('normal','deuteranomaly','protanomaly','tritanomaly'):
        simulated=rgb if name=='normal' else cspace_convert(rgb,dict(name='sRGB1+CVD',cvd_type=name,severity=100),'sRGB1')
        clipped[name]=int(((simulated<0)|(simulated>1)).sum());raw[name]=simulated.tolist()
        cases[name]=cspace_convert(np.clip(simulated,0,1),'sRGB1','CAM02-UCS')
    matrices={n:np.linalg.norm(v[:,None,:]-v[None,:,:],axis=-1) for n,v in cases.items()}
    distances=np.minimum.reduce(list(matrices.values()));chosen=select_maximin(distances,7)
    triple=tuple(chosen[i] for i in select_maximin(distances[np.ix_(chosen,chosen)],3))
    source=RegistryFigures();records=source.test_records();scores={}
    for name in theme.DEFINITIONS:
        values=[r['normalized_cycle_time'] for r in records if r['label']==name]
        if values:scores[name]=float(np.mean(values))
    order=sorted(theme.DEFINITIONS,key=lambda m:(m not in scores,scores.get(m,0.),m));assigned={}
    for method,index in zip([m for m in order if m in theme.TRIO],sorted(triple)):assigned[method]=index
    for method,index in zip([m for m in order if m in theme.CORE and m not in theme.TRIO],sorted(set(chosen)-set(triple))):assigned[method]=index
    remaining=sorted(set(range(10))-set(chosen))
    for i,method in enumerate(m for m in order if m not in theme.CORE):assigned[method]=remaining[i%len(remaining)]
    methods={}
    for name in order:
        methods[name]=dict(base_style=dict(color=colors[assigned[name]],marker=theme.DEFINITIONS[name][0],
            linestyle='-' if '/' in name else '--' if name in ('ilp_tsp','joint_opt') else ':',
            fillstyle='none' if name in ('ilp_tsp','joint_opt') else 'full'),
            names={'tr':theme.DEFINITIONS[name][1],'en':theme.DEFINITIONS[name][2]},palette_index=assigned[name])
    weakest=min((float(distances[i,j]),i,j) for i,j in itertools.combinations(chosen,2))
    _,i,j=weakest;vision=min(matrices,key=lambda n:matrices[n][i,j])
    gray=np.repeat(luminance(rgb)[:,None],3,axis=1)
    gray=np.where(gray<=.0031308,12.92*gray,1.055*gray**(1/2.4)-.055)
    lightness=cspace_convert(gray,'sRGB1','CAM02-UCS')[:,0]
    versions={p:importlib.metadata.version(p) for p in ('matplotlib','colorspacious','numpy')};versions['python']=platform.python_version()
    provenance=git_state()
    result=dict(schema_version=1,library_versions=versions,palette_name='petroff10',palette_hex=colors,palette_rgb=rgb.tolist(),
        vision_cases={k:v.tolist() for k,v in cases.items()},simulated_rgb_before_clipping=raw,
        distances_by_vision={k:v.tolist() for k,v in matrices.items()},minimum_distances=distances.tolist(),clipped_channels=clipped,
        grayscale_lightness=lightness.tolist(),grayscale_pair_differences=np.abs(lightness[:,None]-lightness[None,:]).tolist(),
        white_contrast_ratios=((1.05)/(luminance(rgb)+.05)).tolist(),
        conversion='Interpret original sRGB1 in sRGB1+CVD -> sRGB1 (documented simulation direction), clip [0,1], CAM02-UCS default viewing conditions. Severity 100 is full dichromacy; API names retain anomaly suffixes.',
        grayscale_definition='Linear luminance converted to equal-channel display sRGB; absolute CAM02-UCS lightness differences.',
        subset_count=120,selected_seven=list(chosen),selected_triple=list(triple),
        weakest_selected_pair=dict(indices=[i,j],distance=weakest[0],vision=vision),
        tie_rule='Exact computed maximin score, then lexicographically earliest original palette-index tuple.',
        assignment_rule='Registry test-mean display order within main trio, then remaining core methods, paired with ascending selected indices; others cycle through ascending unused indices.',provenance=provenance)
    definition=dict(schema_version=1,version='v2',palette=dict(name='petroff10',colors=colors,selected_seven=list(chosen),selected_triple=list(triple)),
        methods=methods,roles=theme.DEFAULT_ROLES,colormap=theme.CMAP,font=dict(family='IBM Plex Sans',sizes=dict(report=11,slide=18,title=26,note=18)),
        axes=dict(top=False,right=False,grid='horizontal',grid_alpha=.3),display_order=order,display_scores=scores,
        trio_adjacent=max(order.index(m) for m in theme.TRIO)-min(order.index(m) for m in theme.TRIO)==2,
        measurement_source='palette_check_v1.json',library_versions=versions,provenance=provenance)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    for filename,value in [('palette_check_v1.json',result),('theme_v2.json',definition)]:
        (directory/filename).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    return result,definition
