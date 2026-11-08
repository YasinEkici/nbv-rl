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
PART_MARKERS={'178195':'o','486859':'s','79729':'^'}
MASK_STYLES={'reachable':{'color':'#009E73','marker':'o'},'ik_unreachable':{'color':'#D55E00','marker':'x'},
             'collision_masked':{'color':'#CC79A7','marker':'^'}}
