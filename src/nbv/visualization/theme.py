"""Committed presentation tokens; no runtime palette-measurement dependency."""
from functools import lru_cache
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
CORE=('greedy_cost','mcts','ilp_tsp','joint_opt','sarsa/linear','sarsa/value_estimate','regq/linear')
TRIO=('sarsa/value_estimate','mcts','ilp_tsp')
DEFINITIONS={
 'random':('x','Rastgele','Random'), 'fixed':('*','Sabit sıra','Fixed order'),
 'greedy_ig':('+','Greedy-IG','Greedy IG'), 'greedy_cost':('X','Greedy-IG/maliyet','Greedy IG / cost'),
 'mcts':('h','MCTS','MCTS'), 'ilp_tsp':('>','ILP + TSP','ILP + TSP'),
 'joint_opt':('<','Ortak optimum','Joint optimum'),
 'sarsa/linear':('o','SARSA · Φ doğrusal','SARSA · linear Φ'),
 'sarsa/value_estimate':('P','SARSA · Φ değer tahmini','SARSA · value-estimate Φ'),
 'regq/linear':('s','RegQ','RegQ'), 'myopic/linear':('^','Miyop','Myopic'),
 'sarsa/none':('v','SARSA · Φ yok','SARSA · no shaping'),
 'sarsa/convex':('D','SARSA · Φ dışbükey','SARSA · convex Φ')}
DEFAULT_ROLES={
 'measured':'#0072B2','new':'#E69F00','available':'#56B4E9','unmeasurable':'#969696',
 'reachable':'#009E73','ik_unreachable':'#777777','collision_masked':'#CC79A7',
 'collision':'#D51E1E','context':'#9CA3AB','text':'#20252B','muted':'#555555',
 'background':'#FFFFFF','grid':'#DDDDDD','axis':'#555555','coverage_track':'#DFE5EB',
 'part_mean':'#20252B','split_mean':'#CC79A7','threshold':'#555555','highlight':'#0072B2',
 'table':'#BEC3C9','part_surface':'#AAB7C2','robot':'#737D87','sensor':'#282D32',
 'laser':'#E69F00','unseen':'#999999','table_facing':'#777777'}
CMAP='cividis'

@lru_cache(maxsize=1)
def load_theme():
    path=ROOT/'results/visualization/theme_v2.json'
    if not path.is_file():raise FileNotFoundError('Restore the committed theme v2 artifact.')
    value=json.loads(path.read_text(encoding='utf-8'))
    if value['version']!='v2' or set(value['methods'])!=set(DEFINITIONS):raise ValueError('Invalid theme identity')
    return value

def base_styles():return {m:dict(s['base_style']) for m,s in load_theme()['methods'].items()}
def method_name(method,language='en'):return load_theme()['methods'][method]['names'][language]
def role(name):return load_theme()['roles'][name]
def rgb(name):
    value=role(name).lstrip('#')
    return tuple(int(value[i:i+2],16) for i in (0,2,4))
def rgba(name,alpha):return 'rgba('+','.join(str(v) for v in (*rgb(name),alpha))+')'

def plotly_dash(method):
    """Translate the committed Matplotlib-style token without changing family identity."""
    return {'-':'solid',':':'dot','--':'dash','-.':'dashdot'}[base_styles()[method]['linestyle']]
def display_order(names=None):
    names=set(DEFINITIONS if names is None else names)
    return [m for m in load_theme()['display_order'] if m in names]
def dark_label(method):
    value=base_styles()[method]['color'].lstrip('#')
    return '#'+''.join(f'{round(int(value[i:i+2],16)*.58):02x}' for i in (0,2,4))
