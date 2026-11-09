"""Lossless-payload, portable-font exports for the versioned presentation."""
import copy
import hashlib
import json
from pathlib import Path
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text
from fontTools.ttLib import TTFont
from nbv import evaluation
from nbv.visualization import theme

FONT_ROOT=theme.ROOT/'assets/fonts/ibm-plex-sans'
ALLOWED={'base_style','render_style','rendered_text','title','caption','xlabel','ylabel','notes','note','path','files'}
NEW={'base_style','render_style','rendered_text','render_commit','theme_version','display_order','font'}

def validate_invariance(original,current):
    unexpected=set(current)-set(original)-NEW
    if unexpected:raise ValueError('undeclared field: '+str(sorted(unexpected)))
    if set(original)-set(current):raise ValueError('source payload field removed')
    a={k:v for k,v in original.items() if k not in ALLOWED}
    b={k:current[k] for k in a}
    if json.dumps(a,sort_keys=True,ensure_ascii=False,allow_nan=False)!=json.dumps(b,sort_keys=True,ensure_ascii=False,allow_nan=False):
        raise ValueError('numeric/source payload changed')

def setup_fonts(*,slide=False):
    manifest=FONT_ROOT/'manifest.json'
    if not manifest.is_file():raise FileNotFoundError('Vendored IBM Plex Sans assets are missing')
    m=json.loads(manifest.read_text(encoding='utf-8'))
    for name,sha in m['files'].items():
        p=FONT_ROOT/name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=sha:raise ValueError('Font asset missing or altered: '+name)
        if p.suffix=='.ttf':font_manager.fontManager.addfont(str(p))
    size=theme.load_theme()['font']['sizes']['slide' if slide else 'report']
    plt.rcParams.update({'font.family':theme.load_theme()['font']['family'],'font.size':size,'axes.labelsize':size,
        'axes.titlesize':size+2,'xtick.labelsize':size,'ytick.labelsize':size,'legend.fontsize':size,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'path','axes.unicode_minus':True,
        'figure.facecolor':theme.role('background'),'axes.facecolor':theme.role('background'),
        'text.color':theme.role('text'),'axes.labelcolor':theme.role('text'),'xtick.color':theme.role('axis'),
        'ytick.color':theme.role('axis')})
    return dict(family=theme.load_theme()['font']['family'],versions=m['versions'],files=m['files'],license=m['license'])

def rendered_text(fig,language):
    # Assign explicit font files: no fallback to an installed family or missing glyph.
    for t in fig.findobj(Text):
        weight=t.get_fontweight();name='Bold' if weight in ('bold','heavy',700,800,900) else 'SemiBold' if weight in ('semibold',600) else 'Regular'
        t.set_fontproperties(font_manager.FontProperties(fname=str(FONT_ROOT/f'IBMPlexSans-{name}.ttf'),size=t.get_fontsize()))
    fig.canvas.draw()
    output=[];cmaps={}
    for t in fig.findobj(Text):
        value=t.get_text()
        if not t.get_visible() or not value.strip():continue
        path=font_manager.findfont(t.get_fontproperties(),fallback_to_default=False)
        if not Path(path).resolve().is_relative_to(FONT_ROOT):raise ValueError('Unexpected font fallback')
        if path not in cmaps:cmaps[path]=TTFont(path).getBestCmap()
        missing={c for c in value if not c.isspace() and ord(c) not in cmaps[path]}
        if missing:raise ValueError('Missing font glyphs: '+repr(missing))
        if any(method in value for method in theme.DEFINITIONS):
            raise ValueError('Internal method identifier in rendered text: '+value)
        if language=='tr' and re.search(r'\d+\.\d+',value):raise ValueError('decimal point in Turkish rendered text: '+value)
        output.append(value)
    return output

def display_summary():
    t=theme.load_theme()
    return dict(order=t['display_order'],scores=t['display_scores'],trio_adjacent=t['trio_adjacent'])

def save_figure(fig,target,original,*,language,slide,render_styles=None,title=None,caption=None):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if any(target.with_suffix(ext).exists() for ext in ('.png','.svg','.pdf','.json')):raise FileExistsError(target)
    font=setup_fonts(slide=slide)
    if slide:fig.set_size_inches(12.8,7.2)
    texts=rendered_text(fig,language)
    metadata=copy.deepcopy(original)|dict(base_style=theme.base_styles(),render_style=render_styles or {'mode':'report'},
        rendered_text=texts,render_commit=evaluation.git_state()['commit'],theme_version='v2',
        display_order=theme.display_order(),font=font)
    if title is not None and 'title' in original:metadata['title']=title
    if caption is not None and 'caption' in original:metadata['caption']=caption
    validate_invariance(original,metadata)
    fig.savefig(target.with_suffix('.png'),dpi=150,facecolor=theme.role('background'))
    fig.savefig(target.with_suffix('.pdf'),facecolor=theme.role('background'))
    fig.savefig(target.with_suffix('.svg'),facecolor=theme.role('background'))
    target.with_suffix('.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    plt.close(fig)
    return metadata
