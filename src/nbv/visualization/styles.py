"""Compatibility API backed by the committed global theme, never by a subset."""
from nbv.visualization import theme
PALETTE=tuple(theme.load_theme()['palette']['colors'])
STYLES=theme.base_styles()

def validate_styles(styles):
    triples=[(s['color'],s['marker'],s['linestyle']) for s in styles.values()]
    if len(set(triples))!=len(triples):raise ValueError('duplicate method style triple')
    if set(styles)<=set(theme.CORE) and len({s['color'] for s in styles.values()})!=len(styles):raise ValueError('duplicate core color')

def styles_for(names):
    names=set(names)
    if names-set(STYLES):raise ValueError('unknown method')
    result={n:dict(STYLES[n]) for n in theme.display_order(names)}
    validate_styles(result)
    return result

def scatter_style(style):
    return ({'facecolors':'none','edgecolors':style['color']} if style['fillstyle']=='none' else {'color':style['color']})|{'marker':style['marker']}
PART_MARKERS={'178195':'o','486859':'s','79729':'^'}
MASK_STYLES={name:{'color':theme.role(name),'marker':marker} for name,marker in [('reachable','o'),('ik_unreachable','x'),('collision_masked','^')]}
