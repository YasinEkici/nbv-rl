"""Single Okabe–Ito method dictionary shared by every Plan 004 view and export."""
PALETTE=('#E69F00','#56B4E9','#009E73','#F0E442','#0072B2','#D55E00','#CC79A7','#000000')
_DEFINITIONS={
 'sarsa/linear':('#0072B2','o','-'), 'regq/linear':('#D55E00','s','--'),
 'myopic/linear':('#009E73','^',':'), 'sarsa/none':('#000000','v','-.'),
 'sarsa/convex':('#CC79A7','D','--'), 'sarsa/value_estimate':('#E69F00','P','-'),
 'random':('#56B4E9','x',':'), 'fixed':('#F0E442','*','--'), 'greedy_ig':('#009E73','+','-.'),
 'greedy_cost':('#000000','X','-'), 'ilp_tsp':('#0072B2','>','--'),
 'mcts':('#D55E00','h','-'), 'joint_opt':('#CC79A7','<',':')}
STYLES={name:dict(zip(('color','marker','linestyle'),values)) for name,values in _DEFINITIONS.items()}
for _name,_style in STYLES.items():
    _style['linestyle']='-' if '/' in _name else '--' if _name in ('ilp_tsp','joint_opt') else ':'
    _style['fillstyle']='none' if _name in ('ilp_tsp','joint_opt') else 'full'


def validate_styles(styles):
    triples=[(s['color'],s['marker'],s['linestyle']) for s in styles.values()]
    if len(set(triples))!=len(triples):raise ValueError('duplicate method style triple')
    if len(styles)<=8 and len({s['color'] for s in styles.values()})!=len(styles):raise ValueError('duplicate color in small figure')


def styles_for(names):
    """Stable subset allocation; unique colors <=8, redundant family coding above eight."""
    names=set(names)
    if names-set(STYLES):raise ValueError('unknown method')
    result={n:dict(s) for n,s in STYLES.items() if n in names}
    if len(result)<=8:
        used=set()
        # Retain preferred colors when possible; resolve subset collisions deterministically.
        reserved={s['color'] for s in result.values()}
        for style in result.values():
            if style['color'] in used:
                style['color']=next(c for c in PALETTE if c not in used and c not in reserved)
            used.add(style['color'])
    validate_styles(result)
    return result


def scatter_style(style):
    return ({'facecolors':'none','edgecolors':style['color']} if style['fillstyle']=='none'
            else {'color':style['color']})|{'marker':style['marker']}
PART_MARKERS={'178195':'o','486859':'s','79729':'^'}
MASK_STYLES={'reachable':{'color':'#009E73','marker':'o'},'ik_unreachable':{'color':'#D55E00','marker':'x'},
             'collision_masked':{'color':'#CC79A7','marker':'^'}}
