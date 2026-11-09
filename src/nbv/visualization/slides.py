"""Presentation-only exports from registry/sidecar data; never opens test caches or outputs."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from nbv.visualization.figure_data import RegistryFigures
from nbv.visualization.styles import STYLES
from nbv.visualization.replay import digest
from nbv import evaluation

GREY='#9ca3ab';ACCENT='#E69F00';BLUE='#0072B2'
LABELS={'sarsa/linear':'SARSA / linear','sarsa/value_estimate':'SARSA / value estimate','regq/linear':'RegQ',
        'greedy_cost':'Greedy IG / cost','mcts':'MCTS','ilp_tsp':'ILP + TSP','joint_opt':'Joint optimum',
        'myopic/linear':'Myopic','sarsa/none':'SARSA / none','sarsa/convex':'SARSA / convex'}


def point_style(method,color,*,outline=False):
    hollow=STYLES[method]['fillstyle']=='none'
    return dict(marker=STYLES[method]['marker'],facecolors='none' if hollow else color,
                edgecolors=color if hollow else '#333' if outline else color)


def slide_figure(*,panels=1):
    font='Segoe UI' if any(f.name=='Segoe UI' for f in font_manager.fontManager.ttflist) else 'DejaVu Sans'
    plt.rcParams.update({'font.family':font,'font.size':18,'axes.labelsize':20,'axes.titlesize':23,
        'xtick.labelsize':18,'ytick.labelsize':18,'legend.fontsize':18,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,panels,figsize=(16,9),dpi=120,squeeze=False)
    fig.subplots_adjust(left=.12,right=.94,top=.82,bottom=.22,wspace=.22)
    for ax in axes[0]:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#dddddd',lw=.7);ax.set_axisbelow(True)
    return fig,axes[0][0] if panels==1 else axes[0]


def save_slide(fig,target,metadata):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(target.with_suffix('.png'),dpi=120,facecolor='white')
    fig.savefig(target.with_suffix('.pdf'),facecolor='white')
    target.with_suffix('.json').write_text(json.dumps(metadata|dict(pixels=[1920,1080],aspect='16:9',
        source_policy='Registry/recorded sidecars only; closed test inputs never opened.'),indent=2,allow_nan=False),encoding='utf-8')
    plt.close(fig)


def generate(directory='results/figures/slides',audit='results/collision_audit_val_v1.json'):
    state=evaluation.git_state()
    if state['dirty']:raise ValueError('commit presentation code before final slide generation')
    output=Path(directory);source=RegistryFigures();test=source.test_records();files=[]
    common=dict(schema_version=1,commit=state['commit'],registry_sha256=digest(source.path),
        adaptation='Presentation adaptation of rougier2014; original report figures and sidecars unchanged.',
        style_policy='One highlight (two for quality/latency); other methods grey with their distinct original markers. Direct labels; no upper/right spine; horizontal light grid; minimum 18 pt.',
        method_adaptations='Frozen canonical policies/settings and tail-weight quality records; no new policy evaluation, tuning or selection. Grey context is the user-authorized presentation exception to report colors; original markers/fill/family lines remain.')
    def finish(fig,name,title,caption,*,sources,split,**metadata):
        fig.suptitle(title,x=.12,y=.95,ha='left',fontsize=27,fontweight='semibold')
        fig.text(.12,.075,caption,fontsize=18,va='bottom')
        save_slide(fig,output/name,common|dict(title=title,caption=caption,source_run_ids=sorted(set(sources)),split=split,
            transformations=metadata.pop('transformations',[]))|metadata)
        files.append(name);print('Presentation slide:',name,flush=True)

    # a: test quality from registry only; original VAL-machine timing sidecar stays fixed.
    timing=source.timing();fig,ax=slide_figure()
    ratio=timing['timings']['mcts']['planning_time_ms_per_decision']/timing['timings']['sarsa/value_estimate']['planning_time_ms_per_decision']
    point_data={}
    offsets={'sarsa/linear':22,'regq/linear':-4,'sarsa/value_estimate':-34,'greedy_cost':25,'mcts':-25}
    for method,t in timing['timings'].items():
        records=[e for e in test if e['label']==method];values=[e['normalized_cycle_time'] for e in records]
        x=t['planning_time_ms_per_decision'];color=ACCENT if method=='sarsa/value_estimate' else BLUE if method=='mcts' else GREY
        ax.scatter([x]*len(values),values,s=14,**point_style(method,color),alpha=.17)
        average=float(np.mean(values));ax.scatter([x],[average],s=135,**point_style(method,color,outline=True),zorder=5)
        ax.annotate(LABELS[method],(x,average),xytext=(18,offsets[method]),textcoords='offset points',fontsize=18,
                    arrowprops=dict(arrowstyle='-',color=color,lw=1),bbox=dict(fc='white',ec='none',alpha=.8))
        point_data[method]=dict(ms_per_decision=x,test_mean=average,observations=values)
    ax.set(xscale='log',xlabel='Decision time (ms, log scale)',ylabel='Test normalized cycle time',xlim=(.075,160))
    finish(fig,'p004b_a_quality_latency',f'Measured MCTS / value-estimate decision time: {ratio:.0f}×',
        'Quality: test registry, tail policies, all seeds; dots = observations.\nTime: VAL, seed 0, this machine (D-066). Cross-split descriptive view.',
        sources=[e['source_run'] for e in test if e['label'] in timing['timings']]+[timing['source_run']],
        split=['test quality','VAL timing'],data=point_data,timing=timing)

    # b: all ten actual seed curves, not an interval that might conceal oscillation.
    fig,ax=slide_figure();curves={};runs_used=[]
    for label,color in [('sarsa/linear',ACCENT),('regq/linear',GREY)]:
        runs=source.learning(label);matrix=np.array([r['curve'] for r in runs]);steps=np.array(runs[0]['steps'])/1e6
        for r in runs:ax.plot(steps,r['curve'],color=color,alpha=.16,lw=1.1)
        average=matrix.mean(0);ax.plot(steps,average,color=color,lw=3)
        ax.annotate(LABELS[label],(steps[-1],average[-1]),xytext=(12,10 if label=='sarsa/linear' else -15),textcoords='offset points',fontsize=20,color=color)
        curves[label]=runs;runs_used.extend(r['source_run'] for r in runs)
    s=np.array([r['curve'] for r in curves['sarsa/linear']]);r=np.array([r['curve'] for r in curves['regq/linear']])
    span_s=float(np.median(np.ptp(s,axis=1)));span_r=float(np.median(np.ptp(r,axis=1)))
    ax.set(xlabel='Training steps (millions)',ylabel='VAL normalized cycle time',xlim=(.1,2.45))
    finish(fig,'p004b_b_learning',f'Within-run SARSA span {span_s:.3f}; RegQ span {span_r:.3f}',
        'VAL, ten official seeds. Thin curves = actual runs; thick curves = seed mean.\nSpans = median of each run’s maximum minus minimum over the 20 evaluations.',
        sources=runs_used,split='VAL',curves=curves,median_within_run_span={'sarsa/linear':span_s,'regq/linear':span_r})

    # c: the complete small-sample distribution, retaining baseline observation counts.
    methods=['sarsa/linear','sarsa/value_estimate','greedy_cost','mcts','ilp_tsp'];parts=sorted({e['part'] for e in test},key=str)
    fig,axes=slide_figure(panels=3);dot_data={}
    all_values=[e['normalized_cycle_time'] for e in test if e['label'] in methods]
    lower,upper=min(all_values),max(all_values);padding=(upper-lower)*.04
    for ax,part in zip(axes,parts):
        dot_data[str(part)]={}
        for i,method in enumerate(methods):
            records=[e for e in test if e['part']==part and e['label']==method];values=[e['normalized_cycle_time'] for e in records]
            color=ACCENT if method=='sarsa/value_estimate' else GREY
            jitter=np.linspace(-.15,.15,len(values))
            ax.scatter(values,i+jitter,s=32,**point_style(method,color),alpha=.5)
            ax.scatter([np.mean(values)],[i],s=105,**point_style(method,color,outline=True),lw=.7)
            dot_data[str(part)][method]=dict(records=records,mean=float(np.mean(values)))
        ax.set_title(str(part));ax.set_yticks(range(len(methods)),[LABELS[m] for m in methods] if ax is axes[0] else ['']*len(methods))
        ax.invert_yaxis();ax.set_xlabel('Normalized cycle time')
        ax.set_xlim(lower-padding,upper+padding)
    fig.subplots_adjust(left=.18,right=.97,wspace=.14)
    finish(fig,'p004b_c_test_parts','Test quality varies across parts and individual runs',
        'Test registry, primary tail policies. Every variant × seed observation is shown.\nDeterministic baselines retain their actual point counts; outlined markers = means.',
        sources=[e['source_run'] for e in test if e['label'] in methods],split='test',data=dot_data)

    # c2: the exact frozen paired estimator, with split mean on every part panel.
    paired=source.paired();fig,axes=slide_figure(panels=3)
    differences=[float(np.mean(seeds)) for variants in paired['variants'].values() for seeds in variants.values()]
    lower,upper=min([-.02,0.,*differences]),max([-.02,0.,*differences]);padding=(upper-lower)*.08
    for ax,part in zip(axes,sorted(paired['variants'])):
        variants=paired['variants'][part]
        values=[np.mean(seeds) for seeds in variants.values()]
        ax.scatter(range(1,len(values)+1),values,s=70,color=GREY)
        ax.axhline(0,color='#555',lw=.8);ax.axhline(-.02,color='#555',ls='--',lw=1.5)
        ax.axhline(paired['mean'],color=ACCENT,lw=2.)
        ax.hlines(np.mean(values),2,4,color='#222',lw=3)
        ax.set(title=str(part),xlabel='Variant',xticks=range(1,len(values)+1))
        ax.set_ylim(lower-padding,upper+padding)
    fig.text(.12,.885,'Threshold −0.02 · split mean '+f"{paired['mean']:.4f}",fontsize=18)
    axes[0].set_ylabel('SARSA / linear − myopic')
    finish(fig,'p004b_c2_paired_forward',f"Improvement misses the target by {(.02+paired['mean']):.6f}",
        f"Test, paired variant seed-means. Exact split estimator: {paired['mean']:.15f}.\nOrange = split mean; short black = part mean; dashed = −0.02 threshold. Verdict unchanged.",
        sources=paired['source_runs'],split='test',paired=paired)

    # e: recorded VAL decomposition, with movement highlighted and all movement observations.
    component_path=Path('results/figures/p004_e_val_time_components.json')
    component=json.loads(component_path.read_text(encoding='utf-8'));data=component['component_data']
    methods=['greedy_cost','mcts','ilp_tsp','joint_opt','sarsa/linear','sarsa/value_estimate','regq/linear']
    fig,ax=slide_figure();means={};counts={}
    for key,color,style in [('move',ACCENT,'-'),('scan',GREY,':'),('home','#555','--')]:
        values=[]
        for i,method in enumerate(methods):
            group=[e for e in data if e['method']==method];values.append(float(np.mean([e[key] for e in group])));counts[method]=len(group)
            if key=='move':ax.scatter([i]*len(group),[e[key] for e in group],s=15,**point_style(method,color),alpha=.15)
        means[key]=values;ax.plot(range(len(methods)),values,color=color,ls=style,lw=2.5)
        ax.annotate({'move':'Movement','scan':'Scanning','home':'Return home'}[key],(len(methods)-1,values[-1]),
                    xytext=(12,0),textcoords='offset points',color=color,fontsize=20)
    ax.set_xticks(range(len(methods)),['Greedy','MCTS','ILP+TSP','Joint opt.','SARSA L','SARSA V','RegQ'])
    ax.set(xlim=(-.2,len(methods)+.8),ylabel='Recorded time (s)')
    movement_min=methods[int(np.argmin(means['move']))]
    finish(fig,'p004b_e_val_time_components',f'{LABELS[movement_min]} has the lowest mean VAL movement time',
        'VAL, all 15 variants and recorded canonical seeds. Dots = individual movement times.\nLines = component means; primary tail policies. No test-time decomposition is inferred.',
        sources=[e['run_id'] for e in data if e['method'] in methods],split='VAL',means=means,counts=counts,
        source_sidecar_sha256=digest(component_path))

    # A: per-method/per-part rates from the measured VAL event ledger only.
    collision=json.loads(Path(audit).read_text(encoding='utf-8'))
    if not collision['complete']:raise ValueError('complete VAL collision audit required')
    groups=[r for r in collision['by_part_method'] if r['split']=='val']
    methods=sorted({r['method'] for r in groups});rates={}
    for method in methods:
        selected=[r for r in groups if r['method']==method]
        rates[method]=sum(r['colliding_transfers'] for r in selected)/sum(r['transfers'] for r in selected)
    highlighted=max(methods,key=lambda m:rates[m]);fig,ax=slide_figure()
    for i,method in enumerate(methods):
        color=ACCENT if method==highlighted else GREY
        ax.hlines(i,0,rates[method]*100,colors=color,ls=STYLES[method]['linestyle'],lw=1,alpha=.35)
        ax.scatter([r['colliding_transfer_rate']*100 for r in groups if r['method']==method],[i]*3,
                   s=65,**point_style(method,GREY),alpha=.6)
        ax.scatter([rates[method]*100],[i],s=135,**point_style(method,color,outline=True))
        ax.annotate(f'{rates[method]:.1%}',(rates[method]*100,i),xytext=(12,0),textcoords='offset points',fontsize=18,color='#333')
    ax.set_yticks(range(len(methods)),[LABELS.get(m,m) for m in methods]);ax.invert_yaxis()
    ax.set(xlabel='Colliding transfers (%)',xlim=(-2,max(rates.values())*100+17))
    fig.subplots_adjust(left=.22)
    finish(fig,'p004b_collision_rates',f'Unchecked paths collide in {rates[highlighted]:.1%} of {LABELS.get(highlighted,highlighted)} transfers',
        'VAL recorded paths, unchanged D-042 geometry, maximum 1° joint sampling.\nFaint = individual parts; outlined = pooled transfer rate, including home. Descriptive only.',
        sources=list(collision['sources']['sources']),split='VAL',rates=rates,by_part=groups,
        audit_sha256=digest(audit),highlight_rule='Largest measured transfer rate; visual emphasis only, no policy/setting selection.')
    index=['# Presentation figures (Plan 004b)', '',
        'Separate presentation exports. Original report figures and the immutable demo are unchanged. '
        'All uncertainty/distribution encodings and data splits are stated on the figures and in their JSON receipts.', '',
        'Presentation principles adapted from [Rougier, Droettboom & Bourne (2014)](https://doi.org/10.1371/journal.pcbi.1003833); '
        'the exact layout and typography are project choices.', '',
        '| Figure | PNG | PDF | Sources and transformations |','|---|---|---|---|']
    manifest={}
    for name in files:
        metadata=json.loads((output/f'{name}.json').read_text(encoding='utf-8'))
        index.append(f"| {metadata['title']} | [PNG]({name}.png) | [PDF]({name}.pdf) | [JSON]({name}.json) |")
        manifest[name]={suffix:digest(output/f'{name}.{suffix}') for suffix in ('png','pdf','json')}
    (output/'README.md').write_text('\n'.join(index)+'\n',encoding='utf-8')
    (output/'manifest_v1.json').write_text(json.dumps(dict(commit=state['commit'],files=manifest),indent=2),encoding='utf-8')
    return files
