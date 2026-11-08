"""Plan 004 static views. Test quality inputs come exclusively from registry JSON."""
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from nbv import parts
from nbv.visualization.assets import load_scene
from nbv.visualization.figure_data import RegistryFigures, rank_runs
from nbv.visualization.replay import digest, validated_records
from nbv.visualization.styles import STYLES, PART_MARKERS, MASK_STYLES,styles_for,scatter_style
from nbv.evaluation import git_state

DEMO_METHODS=('greedy_cost','mcts','ilp_tsp','joint_opt','sarsa/linear','sarsa/value_estimate','regq/linear')


def draw_paired_reference(ax,split_mean,part_mean):
    return {'split':ax.axhline(split_mean,color='#CC79A7',ls='-.',lw=1.5,label='split mean'),
            'part':ax.hlines(part_mean,1.35,2.65,color='#0072B2',lw=3,label='part mean')}


def candidate_angles(direction):
    return np.degrees(np.arctan2(direction[:,1],direction[:,0])),np.degrees(np.arcsin(np.clip(direction[:,2],-1,1)))


def gallery_range(scenes):
    counts=[s['nominal_measured'][s['plan']['reachable'].astype(bool)].sum(0) for s in scenes]
    positive=np.concatenate([c[c>0] for c in counts])
    return int(positive.min()),int(positive.max())


def gallery_credit(scene):
    source=scene['attribution'];family='CC BY-SA' if source['share_alike'] else 'CC BY'
    return (f"{scene['part']}: "+'\n'.join(textwrap.wrap(source['name'],34))+
            f"\n{source['author']} · {family} (version unknown)\n{source['thing_url']}")


def generate(directory='results/figures'):
    state=git_state()
    if state['dirty']: raise ValueError('commit renderer before producing figures')
    output=Path(directory); output.mkdir(parents=True,exist_ok=True)
    registry=RegistryFigures(); test=registry.test_records()
    replay,validation=validated_records()
    common=dict(schema_version=1,commit=state['commit'],registry_sha256=digest(registry.path),
        styles=STYLES,intervals='Individual observations; no uncertainty interval unless explicitly labeled.',
        test_access='Registry metrics only; no test cache/output/environment access.',
        method_adaptations={
            'all':'Frozen canonical methods/settings (D-041/D-053), no tuning, policy selection or changed scientific claim.',
            'learners':'Primary D-046 tail-average policies; method/Phi labels distinguish the original trained weights, eleven D-049 features.',
            'baselines':'Original nominal-model planners and recorded repair; actual stochastic seed counts retained.',
            'mcts':'Original nominal rollout/search budget and saved seed identities; no fresh search for these views.',
            'replay':'Stored sequence only; full oracle A then partial oracle B (D-069), no policy calls.',
            'learning_highlights':'Adapted from tanaka2026: ten runs, final official-policy score, actual best/lower-median/worst; rank line styles override method line styles in this panel only.'})
    files=[];active_styles=styles_for(list(STYLES))

    def save(fig,name,*,split,sources=(),transforms=(),**extra):
        fig.savefig(output/f'{name}.png',dpi=170,bbox_inches='tight')
        fig.savefig(output/f'{name}.pdf',bbox_inches='tight')
        metadata=common|dict(styles=active_styles,split=split,source_run_ids=sorted(set(sources)),
            source_commits={r['run_id']:r['commit'] for r in registry.rows if r['run_id'] in sources},
            transformations=list(transforms))|extra
        (output/f'{name}.json').write_text(json.dumps(metadata,indent=2,allow_nan=False),encoding='utf-8')
        plt.close(fig); files.extend(f'{name}.{suffix}' for suffix in ('png','pdf','json'))
        print(f'Figure: {name}',flush=True)

    def dot(ax,x,y,label,**kw):
        style=active_styles[label]
        ax.scatter(x,y,**scatter_style(style),label=label,**kw)

    timing=registry.timing()
    active_styles=styles_for(timing['timings'])
    fig,ax=plt.subplots(figsize=(10,6))
    for label,t in timing['timings'].items():
        group=[e for e in test if e['label']==label]
        ys=[e['normalized_cycle_time'] for e in group]; x=t['planning_time_ms_per_decision']
        dot(ax,[x]*len(ys),ys,label,alpha=.15,s=17)
        ax.scatter([x],[np.mean(ys)],**scatter_style(active_styles[label]),s=110)
    ax.set(xscale='log',xlabel='Planning time (ms / decision; log)',ylabel='Test normalized cycle time',title='Quality and decision time: descriptive cross-split view')
    ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.text(.5,.005,'Time: VAL, seed 0, this machine (D-066); quality: test, tail policies, all seeds. Faint = observations; large = mean.',ha='center',fontsize=8)
    save(fig,'p004_a_quality_latency',split=['test quality','val timing'],sources=[e['source_run'] for e in test if e['label'] in timing['timings']]+[timing['source_run']],
         transforms=['No Pareto optimality inference across splits; exact registered timing reused.'],timing=timing)

    labels=('sarsa/linear','regq/linear','sarsa/value_estimate')
    active_styles=styles_for(labels)
    fig,axes=plt.subplots(1,3,figsize=(15,5),sharey=True); rankings={}; sources=[]
    for ax,label in zip(axes,labels):
        runs=registry.learning(label); ranking=rank_runs(runs); rankings[label]=ranking
        for r in runs:
            ax.plot(np.array(r['steps'])/1e6,r['curve'],color=active_styles[label]['color'],alpha=.18,lw=1)
            sources.append(r['source_run'])
        for name,line in [('best','-'),('median','-.'),('worst','--')]:
            r=next(r for r in runs if r['seed']==ranking[name])
            ax.plot(np.array(r['steps'])/1e6,r['curve'],color=active_styles[label]['color'],linestyle=line,lw=2,label=f'{name}: seed {r["seed"]}')
        ax.set(title=label,xlabel='Training steps (millions)'); ax.legend(fontsize=8);ax.grid(alpha=.2)
    axes[0].set_ylabel('VAL normalized cycle time')
    fig.suptitle('Ten individual VAL learning curves; actual best / lower-median / worst seeds')
    fig.text(.5,.005,'No bands or percentiles. Ranking: official tail-policy VAL score; ties: seed; lower median: fifth sorted run. Adapted from Tanaka & Mahmood (2026).',ha='center',fontsize=8)
    save(fig,'p004_b_learning_runs',split='val',sources=sources,transforms=['Original registry history, 20 points per seed; no smoothing.'],rankings=rankings)

    labels=list(STYLES); part_ids=sorted({str(e['part']) for e in test})
    active_styles=styles_for(labels)
    fig,axes=plt.subplots(1,3,figsize=(17,8),sharey=True); counts={}
    for ax,fid in zip(axes,part_ids):
        for i,label in enumerate(labels):
            group=[e for e in test if str(e['part'])==fid and e['label']==label]
            # Deterministic offsets expose coincident observations without inventing seeds.
            offset=np.linspace(-.22,.22,len(group)) if len(group)>1 else [0]
            dot(ax,i+np.asarray(offset),[e['normalized_cycle_time'] for e in group],label,alpha=.35,s=18)
            counts[f'{fid}/{label}']=len(group)
        ax.set(title=f'Test part {fid}',xticks=range(len(labels)),xticklabels=labels)
        ax.tick_params(axis='x',labelrotation=80);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Normalized cycle time');fig.suptitle('Test: every registered variant × seed observation')
    fig.text(.5,.005,'Horizontal offsets only separate points; deterministic baselines retain their actual 5 observations per part. No intervals.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.96))
    save(fig,'p004_c_test_distributions',split='test',sources=[e['source_run'] for e in test],transforms=['Deterministic display offsets; all original observations preserved.'],observation_counts=counts)

    paired=registry.paired(); fig,axes=plt.subplots(1,3,figsize=(13,5),sharey=True)
    for ax,(fid,variants) in zip(axes,paired['variants'].items()):
        values=[float(np.mean(v)) for v in variants.values()]
        dot(ax,range(len(values)),values,'sarsa/linear',s=45)
        ax.axhline(0,color='#000000',lw=.8,label='zero');ax.axhline(-.02,color='#D55E00',ls='--',label='predeclared -0.02')
        draw_paired_reference(ax,paired['mean'],float(np.mean(values)))
        ax.set(title=f'Test part {fid}',xticks=range(5),xticklabels=[v.split('_')[-1] for v in variants],xlabel='Variant');ax.grid(alpha=.15)
    axes[0].set_ylabel('SARSA/linear − myopic normalized cycle time')
    axes[-1].legend(fontsize=8); fig.suptitle('Paired differences after averaging the same ten seeds within each variant')
    fig.text(.5,.005,f'Split mean = {paired["mean"]:.15f}; individual variant differences, no significance test or uncertainty band.',ha='center',fontsize=9)
    save(fig,'p004_c2_paired_forward',split='test',sources=paired['source_runs'],transforms=['D-055 paired_difference, then paired seed mean per variant; nested part/variant split mean.'],paired=paired)

    demo=[r for r in replay if r['variant']=='271875_v0' and r['seed']==0 and r['method'] in DEMO_METHODS]
    if len(demo)!=7: raise ValueError('seven original seed-0 demo episodes required')
    active_styles=styles_for(DEMO_METHODS)
    fig,ax=plt.subplots(figsize=(11,6))
    for order,method in enumerate(DEMO_METHODS):
        r=next(r for r in demo if r['method']==method)
        style=active_styles[r['method']]; times=[t['elapsed_before_home'] for t in r['joint_trace']]; cov=[s['coverage'] for s in r['steps']]
        ax.step([0]+times,[0]+cov,where='post',label=r['method'],**style,lw=1.6,markersize=3,zorder=order+2)
        ax.plot([times[-1],r['summary']['cycle_time']],[cov[-1]]*2,color=style['color'],ls=':',lw=2)
        ax.scatter([r['summary']['cycle_time']],[cov[-1]],**scatter_style(style),s=35,zorder=order+2)
    ax.axhline(.95,color='#000000',ls='--',lw=.8);ax.set(xlabel='Accumulated cycle time (s)',ylabel='Primary coverage',title='VAL 271875_v0: stored sequences, seed 0');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.text(.5,.005,'Dotted final segment = home. Draw order (later on top): greedy, MCTS, ILP, joint, SARSA/linear, SARSA/value, RegQ.\nLearner coverage derived from replay, consistent with partial oracle; baseline steps match full oracle.',ha='center',fontsize=8)
    save(fig,'p004_d_val_coverage_time',split='val',sources=[r['source_run'] for r in demo],transforms=['Exact stored-sequence replay; no policy calls.'],draw_order=list(DEMO_METHODS),overlap='Later methods on top, including shared greedy/SARSA-value prefix.',replay_manifest_sha256=digest('results/visualization/replay_validation_v1.json'),verification={r['method']:r['verification'] for r in demo})

    val_labels=[label for label in labels if any(r['method']==label for r in replay)]
    active_styles=styles_for(val_labels)
    fig,axes=plt.subplots(1,3,figsize=(16,6)); component_data=[]
    for r in replay:
        component_data.append(dict(run_id=r['source_run'],variant=r['variant'],method=r['method'],seed=r['seed'],
            move=sum(s['t_move'] for s in r['steps']),scan=sum(s['t_scan'] for s in r['steps']),home=sum(s['t_home'] for s in r['steps'])))
    for ax,component in zip(axes,('move','scan','home')):
        for i,label in enumerate(val_labels):
            group=[r for r in component_data if r['method']==label]
            dot(ax,i+np.linspace(-.25,.25,len(group)),[r[component] for r in group],label,s=12,alpha=.3)
        ax.set(title=f'VAL: {component}',ylabel='Time (s)',xticks=range(len(val_labels)),xticklabels=val_labels);ax.tick_params(axis='x',labelrotation=80);ax.grid(axis='y',alpha=.2)
    fig.suptitle('VAL time decomposition: 15 variants, all canonical learner seeds and stored baseline runs')
    fig.tight_layout(rect=(0,0,1,.94))
    save(fig,'p004_e_val_time_components',split='val',sources=[r['source_run'] for r in replay],transforms=['Separate move/scan/home sums of exact replay steps; individual observations, no intervals.'],component_data=component_data)

    active_styles=styles_for(labels)
    fig,ax=plt.subplots(figsize=(11,7))
    for label in labels:
        for fid in part_ids:
            group=[r for r in test if r['label']==label and str(r['part'])==fid]
            ax.scatter([r['total_joint_motion'] for r in group],[r['normalized_cycle_time'] for r in group],
                       color=active_styles[label]['color'],marker=PART_MARKERS[fid],s=13,alpha=.12)
            ax.scatter([np.mean([r['total_joint_motion'] for r in group])],[np.mean([r['normalized_cycle_time'] for r in group])],
                       **scatter_style(active_styles[label]),s=70,label=label if fid==part_ids[0] else None)
    ax.set(xlabel='Total joint motion (rad)',ylabel='Normalized cycle time',title='Test: recorded joint motion and cycle time; no component-time reconstruction');ax.legend(fontsize=7,ncol=2);ax.grid(alpha=.2)
    fig.text(.5,.005,'Faint points: part markers '+', '.join(f'{p}={PART_MARKERS[p]}' for p in part_ids)+'; large method-style points: mean within each part. No intervals.',ha='center',fontsize=8)
    save(fig,'p004_e_test_motion_time',split='test',sources=[r['source_run'] for r in test],transforms=['Only registered total motion and normalized cycle time; part means additionally shown.'],part_markers=PART_MARKERS)

    manifest=parts.load_yaml('configs/split_v1.yaml')
    scenes={split:[load_scene(p['file_id']) for p in manifest['parts'] if p['split']==split] for split in ('val','train')}
    for split,group in scenes.items():
        vmin,vmax=gallery_range(group)
        ncols=min(4,len(group)); nrows=(len(group)+ncols-1)//ncols
        fig=plt.figure(figsize=(4.6*ncols,5*nrows+2)); geometry_sources={};licenses={}
        for i,scene in enumerate(group):
            ax=fig.add_subplot(nrows,ncols,i+1,projection='3d')
            points=scene['points']; count=scene['nominal_measured'][scene['plan']['reachable'].astype(bool)].sum(0)
            table=scene['normals'][:,2]<=-np.cos(np.deg2rad(10)); seen=(count>0)&~table; unseen=(count==0)&~table
            artist=ax.scatter(*points[seen].T,c=count[seen],s=.22,cmap='cividis',vmin=vmin,vmax=vmax,rasterized=True)
            ax.scatter(*points[unseen].T,s=.1,c='#999999',alpha=.24,marker='x',rasterized=True)
            ax.scatter(*points[table].T,s=.12,c='#777777',alpha=.24,marker='v',rasterized=True)
            ax.set(xlabel='x (mm)',ylabel='y (mm)',zlabel='z (mm)',box_aspect=(1,1,1))
            ax.set_title(gallery_credit(scene),fontsize=8,pad=12)
            geometry_sources.update(scene['source_files']);licenses[str(scene['part'])]=scene['attribution']|{
                'license_version':'version not recorded','official_specific_license_url':None,
                'official_license_overview':'https://creativecommons.org/licenses/',
                'changes':['Scale nominal mesh to 200 mm longest side','20,000 area-weighted nominal surface samples','Derived nominal visibility/pass counts'],
                'share_alike_derivative':bool(scene['attribution']['share_alike'])}
        fig.colorbar(artist,ax=fig.axes,shrink=.45,pad=.03,label=f'Reachable measured passes (actual positive range {vmin}–{vmax})')
        fig.suptitle(f'{split.upper()}: nominal surface accessibility (20,000 samples per part)')
        fig.text(.5,.04,'Faint gray triangles: table-facing n_z ≤ −cos(10°); faint crosses: other never-seen samples. Part frame: bbox bottom-center, z up; no new visibility computation.',ha='center',fontsize=8)
        fig.text(.5,.012,'Thingi10K; changes: 200 mm scaling, area-weighted samples, derived visibility counts. License overview: creativecommons.org/licenses/; version-specific deed unknown.',ha='center',fontsize=8)
        save(fig,f'p004_f_{split}_gallery',split=split,transforms=['Count nominal measured bits over saved reachable passes; table-facing takes display priority.'],color_scale={'map':'cividis','vmin':vmin,'vmax':vmax},source_files=geometry_sources,attribution=licenses)

        fig=plt.figure(figsize=(4*ncols,3.6*nrows+1))
        for i,scene in enumerate(group):
            ax=fig.add_subplot(nrows,ncols,i+1); plan=scene['plan']; direction=plan['direction']
            azimuth,elevation=candidate_angles(direction)
            masks={'reachable':plan['reachable'].astype(bool),'ik_unreachable':~plan['ik_reachable'].astype(bool),
                'collision_masked':plan['ik_reachable'].astype(bool)&~plan['reachable'].astype(bool)}
            for reason,mask in masks.items():
                style=MASK_STYLES[reason];ax.scatter(azimuth[mask],elevation[mask],c=style['color'],marker=style['marker'],s=24,label=f'{reason}: {int(mask.sum())}')
            ax.set(xlabel='Azimuth (deg)',ylabel='Elevation (deg)',title=str(scene['part']),xlim=(-180,180),ylim=(0,90));ax.legend(fontsize=6,loc='upper left');ax.grid(alpha=.2)
        fig.suptitle(f'{split.upper()}: 48 saved directions, azimuth–elevation mask map')
        fig.text(.5,.025,'Part frame: azimuth atan2(y,x), elevation asin(z), degrees; z upward. Saved IK/collision masks; no new IK.',ha='center',fontsize=9)
        save(fig,f'p004_g_{split}_candidate_sphere',split=split,transforms=['2D azimuth/elevation of saved unit directions; original IK and collision masks; legacy filename retained.'],source_files=geometry_sources)
    result=dict(commit=state['commit'],files=files,figure_count=len(files)//3,replay_validation=validation['counts'],paired_mean=paired['mean'])
    Path('results/visualization/static_figures_v1.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result
