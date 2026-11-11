"""Theme-v2 figures from frozen registry and v1 sidecars; no policy calls."""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import textwrap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from nbv.visualization import theme,assets
from nbv.visualization import prepare_export_workspace
from nbv.visualization.figure_data import RegistryFigures,rank_runs
from nbv.visualization.replay import validated_records
from nbv.visualization.design_export import setup_fonts,save_figure,validate_invariance

def number(value,lang,digits=3):
    s=f'{value:.{digits}f}'
    return (s.replace('.',',') if lang=='tr' else s).replace('-','−')

def tick(value,position,lang):
    text=f'{value:.3g}'
    return (text.replace('.',',') if lang=='tr' else text).replace('-','−')

def color(method,stage,highlight):
    return theme.base_styles()[method]['color'] if stage is None or stage=='b' and method in highlight else theme.role('context')

def points(ax,x,y,method,c,**kw):
    s=theme.base_styles()[method]
    kwargs={'marker':s['marker'],'facecolors':'none','edgecolors':c} if s['fillstyle']=='none' else {'marker':s['marker'],'color':c}
    return ax.scatter(x,y,**kwargs,**kw)

def make_figure(panels,slide):
    setup_fonts(slide=slide)
    fig,axes=plt.subplots(1,panels,figsize=(12.8,7.2) if slide else (max(11,panels*5),8),squeeze=False)
    fig.subplots_adjust(left=.23 if panels==1 else .19,right=.94,top=.84 if slide else .78,bottom=.30 if slide else .25,wspace=.24)
    return fig,list(axes[0])

def decorate(fig,axes,title,caption,lang,slide):
    for ax in axes:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color=theme.role('grid'),alpha=.5);ax.set_axisbelow(True)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p:tick(x,p,lang)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x,p:tick(x,p,lang)))
    fig.suptitle(title,x=.12,y=.96,ha='left',fontsize=26 if slide else 19,fontweight='semibold')
    fig.text(.12,.045,'\n'.join(textwrap.wrap(caption,80 if slide else 145)),ha='left',va='bottom',fontsize=18 if slide else 10)

def draw(kind,old,lang,stage=None):
    slide=stage is not None;tr=lang=='tr';source=RegistryFigures();test=source.test_records()
    selected={'a':{'sarsa/value_estimate','mcts'},'b':{'sarsa/linear'},'c':{'sarsa/value_estimate'},
              'c2':{'sarsa/linear'},'e':set(theme.DEFINITIONS),'collision':{'sarsa/convex'}}.get(kind,set())
    styles={m:dict(mode='report' if not slide else 'highlight' if stage=='b' and m in selected else 'context_gray',
                   color=color(m,stage,selected),label_color=theme.dark_label(m) if not slide or stage=='b' and m in selected else theme.role('muted')) for m in theme.DEFINITIONS}
    title='';caption='';named_axes=[]
    if kind=='a':
        fig,axes=make_figure(1,slide);ax=axes[0];timing=source.timing()['timings']
        for i,m in enumerate(theme.display_order(timing)):
            values=[r['normalized_cycle_time'] for r in test if r['label']==m];x=timing[m]['planning_time_ms_per_decision'];c=styles[m]['color']
            points(ax,[x]*len(values),values,m,c,s=18,alpha=.18)
            y=float(np.mean(values));points(ax,[x],[y],m,c,s=95)
            ax.annotate(theme.method_name(m,lang),(x,y),xytext=(8,12 if i%2 else -22),textcoords='offset points',
                        color=styles[m]['label_color'],fontsize=18 if slide else 11)
        ax.set_xscale('log');ax.set_xlabel('Karar başına süre (ms, log)' if tr else 'Time per decision (ms, log)')
        ax.set_ylabel('Normalize çevrim süresi' if tr else 'Normalized cycle time')
        ratio=timing['mcts']['planning_time_ms_per_decision']/timing['sarsa/value_estimate']['planning_time_ms_per_decision']
        title=(f'Benzer kalite, karar süresinde {number(ratio,lang,0)} kat fark' if tr else f'Similar quality, {number(ratio,lang,0)}-fold decision-time difference')
        caption='Süre: VAL, seed 0, bu makine. Kalite: test, bütün seedler, kuyruk politikası. Noktalar tek gözlemler; bölmeler arası betimsel görünüm.' if tr else 'Time: VAL, seed 0, this machine. Quality: test, all seeds, tail policies. Individual observations; descriptive cross-split comparison.'
    elif kind=='b':
        methods=['sarsa/linear','regq/linear'] if slide else ['sarsa/linear','regq/linear','sarsa/value_estimate']
        methods=theme.display_order(methods);fig,axes=make_figure(1 if slide else len(methods),slide)
        for m in methods:
            ax=axes[0] if slide else axes[methods.index(m)];runs=source.learning(m);matrix=np.array([r['curve'] for r in runs]);x=np.array(runs[0]['steps'])/1e6;c=styles[m]['color']
            for r in runs:ax.plot(x,r['curve'],color=c,alpha=.19,lw=.8)
            if slide:
                y=matrix.mean(0);ax.plot(x,y,color=c,lw=2.8)
                ax.annotate(theme.method_name(m,lang),(x[-1],y[-1]),xytext=(10,12 if m=='sarsa/linear' else -18),textcoords='offset points',color=styles[m]['label_color'])
            else:
                rank=rank_runs(runs)
                for label,ls in [('best','-'),('median','-.'),('worst','--')]:
                    r=next(r for r in runs if r['seed']==rank[label]);ax.plot(x,r['curve'],color=c,ls=ls,lw=1.8,label=f'{label}: seed {r["seed"]}')
                ax.set_title(theme.method_name(m,lang));ax.legend()
            ax.set_xlabel('Eğitim adımı (milyon)' if tr else 'Training steps (millions)');ax.set_ylabel('VAL normalize çevrim süresi' if tr else 'VAL normalized cycle time')
        if slide:axes[0].set_xlim(.1,2.7)
        spans={m:float(np.median(np.ptp(np.array([r['curve'] for r in source.learning(m)]),axis=1))) for m in ('sarsa/linear','regq/linear')}
        title=(f'SARSA salınımı {number(spans["sarsa/linear"],lang)}; RegQ {number(spans["regq/linear"],lang)}' if tr else 'Individual runs expose linear-policy variation')
        caption='VAL, 10 seed, 20 değerlendirme. İnce çizgiler tek koşular, kalın çizgi seed ortalaması. Sayılar koşu içi açıklıkların medyanıdır.' if tr else 'VAL, ten actual runs, twenty evaluations. Thin curves are individual runs; best/lower-median/worst are actual selected runs, not percentile bands.'
    elif kind=='c':
        methods=theme.display_order(['sarsa/linear','sarsa/value_estimate','greedy_cost','mcts','ilp_tsp'] if slide else None)
        parts=sorted({r['part'] for r in test},key=str);fig,axes=make_figure(3,slide)
        values=[r['normalized_cycle_time'] for r in test if r['label'] in methods];lo,hi=min(values),max(values);pad=(hi-lo)*.05
        for ax,fid in zip(axes,parts):
            for i,m in enumerate(methods):
                ys=[r['normalized_cycle_time'] for r in test if r['part']==fid and r['label']==m]
                points(ax,ys,i+np.linspace(-.17,.17,len(ys)),m,styles[m]['color'],s=19 if slide else 10,alpha=.4)
                points(ax,[np.mean(ys)],[i],m,styles[m]['color'],s=80)
            ax.set_title(str(fid));ax.set_xlim(lo-pad,hi+pad);ax.set_xlabel('Normalize süre' if tr else 'Normalized cycle time')
            ax.set_yticks(range(len(methods)));named_axes.append((ax,[theme.method_name(m,lang) if ax is axes[0] else '' for m in methods]))
            ax.invert_yaxis()
        title='Parça türü performans dağılımını değiştiriyor' if tr else 'Test quality varies across parts and individual observations'
        caption='Test, kuyruk politikaları. Bütün varyant × seed noktaları korunur; büyük işaretler ortalama. Deterministik baseline gözlem sayıları artırılmaz.' if tr else 'Test, tail policies. All original variant-by-seed observations; large markers are means. Deterministic baseline counts are retained.'
    elif kind=='c2':
        paired=source.paired();fig,axes=make_figure(3,slide);all_values=[]
        for ax,fid in zip(axes,sorted(paired['variants'])):
            values=[float(np.mean(v)) for v in paired['variants'][fid].values()];all_values+=values
            ax.scatter(range(1,6),values,color=styles['sarsa/linear']['color'],s=55)
            ax.axhline(0,color=theme.role('axis'),lw=.8);ax.axhline(-.02,color=theme.role('threshold'),ls='--',lw=1.2)
            ax.axhline(paired['mean'],color=styles['sarsa/linear']['color'],lw=2)
            ax.hlines(np.mean(values),2,4,color=theme.role('part_mean'),lw=2)
            ax.set_title(str(fid));ax.set_xlabel('Varyant' if tr else 'Variant');ax.set_xticks(range(1,6))
        for ax in axes:ax.set_ylim(min(all_values)-.01,max(all_values)+.01)
        axes[0].set_ylabel('SARSA − miyop farkı' if tr else 'SARSA minus myopic difference')
        title=f'İyileşme {number(paired["mean"],lang,4)}; eşik −0,02' if tr else f'Paired improvement {number(paired["mean"],lang,4)} misses the threshold'
        caption='Test; her nokta bir varyantın eşleştirilmiş seed ortalaması. Uzun renkli çizgi bölme, kısa koyu çizgi parça ortalaması; kesikli çizgi eşik. Karar değişmedi.' if tr else 'Test; paired seed means within variants. Long colored line: split mean; short dark line: part mean; dashed line: threshold. Verdict unchanged.'
    elif kind=='e':
        values=json.loads(Path('results/figures/p004_e_val_time_components.json').read_text(encoding='utf-8'))['component_data']
        methods=theme.display_order({r['method'] for r in values});fig,axes=make_figure(1,slide);ax=axes[0]
        for i,m in enumerate(methods):
            group=[r for r in values if r['method']==m];left=0.
            for component,hatch in [('move',''),('scan','///'),('home','...')]:
                value=float(np.mean([r[component] for r in group]));c=styles[m]['color'] if component=='move' else theme.role('context')
                ax.barh(i,value,left=left,color=c,hatch=hatch,edgecolor=theme.role('background'),height=.65)
                if i==len(methods)-1:ax.text(left+value/2,1.02,{'move':'Hareket','scan':'Tarama','home':'Dönüş'}[component] if tr else component,transform=ax.get_xaxis_transform(),ha='center',va='bottom',fontsize=18 if slide else 10)
                left+=value
        ax.set_yticks(range(len(methods)));named_axes.append((ax,[theme.method_name(m,lang) for m in methods]));ax.invert_yaxis()
        ax.set_xlabel('Ortalama süre (s)' if tr else 'Mean time (s)')
        title='VAL çevriminde hareket maliyeti yöntemleri ayırıyor' if tr else 'Movement contributes to the VAL cycle-time gap'
        caption='VAL, 15 varyant; bütün kayıtlı baseline ve kuyruk politikası bölümleri. Düz bölge hareket, taralı bölge tarama, noktalı bölge eve dönüş. Ortalamalar betimseldir.' if tr else 'VAL, all 15 variants and recorded primary episodes. Solid: movement; hatching: scanning; dots: return home. Descriptive means.'
    elif kind=='collision':
        rates=old['rates'];methods=theme.display_order(rates);fig,axes=make_figure(1,slide);ax=axes[0]
        for i,m in enumerate(methods):
            per_part=[r['colliding_transfer_rate']*100 for r in old['by_part'] if r['method']==m]
            points(ax,per_part,[i]*len(per_part),m,theme.role('context'),s=35,alpha=.5)
            points(ax,[rates[m]*100],[i],m,styles[m]['color'],s=95)
            ax.annotate(number(rates[m]*100,lang,1)+'%',(rates[m]*100,i),xytext=(8,0),textcoords='offset points',va='center')
        ax.set_yticks(range(len(methods)));named_axes.append((ax,[theme.method_name(m,lang) for m in methods]));ax.invert_yaxis()
        ax.set_xlim(-1,max(rates.values())*100+7);ax.set_xlabel('Çarpışmalı geçiş (%)' if tr else 'Colliding transfers (%)')
        title='Denetlenmeyen geçişler çarpışma içeriyor' if tr else 'Unchecked recorded transfers contain collisions'
        caption='VAL, değişmeyen geometri, en fazla 1° eklem örnekleme. Küçük işaretler parçalar; büyük işaret eve dönüş dahil toplam oran. Sürekli güvenlik garantisi değildir.' if tr else 'VAL, frozen geometry, maximum 1-degree sampling. Small markers: parts; large markers: pooled rate including home. No continuous-safety guarantee.'
    elif kind=='d':
        replay,_=validated_records();selected_records=[r for r in replay if r['variant']=='271875_v0' and r['seed']==0 and r['method'] in theme.CORE]
        fig,axes=make_figure(1,slide);ax=axes[0]
        for m in theme.display_order(r['method'] for r in selected_records):
            r=next(r for r in selected_records if r['method']==m);s=theme.base_styles()[m]
            x=[0.]+[j['elapsed_before_home'] for j in r['joint_trace']];y=[0.]+[v['coverage'] for v in r['steps']]
            ax.step(x,y,where='post',color=s['color'],ls=s['linestyle'],label=theme.method_name(m,lang),lw=1.6)
            ax.plot([x[-1],r['summary']['cycle_time']],[y[-1]]*2,color=s['color'],ls=':')
        ax.axhline(.95,color=theme.role('threshold'),ls='--');ax.legend(ncol=2)
        ax.set(xlabel='Accumulated cycle time (s)',ylabel='Primary coverage');title='Stored VAL trajectories, including return home'
        caption='VAL 271875, variant 0, seed 0. Baseline steps match the full oracle; learner coverage is derived from replay and consistent with partial summaries.'
    elif kind=='motion':
        fig,axes=make_figure(1,slide);ax=axes[0]
        for m in theme.display_order():
            group=[r for r in test if r['label']==m]
            points(ax,[r['total_joint_motion'] for r in group],[r['normalized_cycle_time'] for r in group],m,theme.base_styles()[m]['color'],s=12,alpha=.2,label=theme.method_name(m,lang))
        ax.set(xlabel='Total joint motion (rad)',ylabel='Normalized cycle time');ax.legend(ncol=2)
        title='Test joint movement and recorded cycle time';caption='Registry observations only; no test geometry or time decomposition. Individual variant-by-seed points, primary policies.'
    else:
        split='train' if 'train' in kind else 'val';gallery=kind.startswith('f_')
        manifest=__import__('nbv.parts',fromlist=['load_yaml']).load_yaml('configs/split_v1.yaml')
        scenes=[assets.load_scene(p['file_id']) for p in manifest['parts'] if p['split']==split]
        setup_fonts(slide=False);fig=plt.figure(figsize=(18,11 if split=='train' else 7));axes=[]
        for i,s in enumerate(scenes):
            ax=fig.add_subplot(2 if split=='train' else 1,4 if split=='train' else 3,i+1,projection='3d' if gallery else None);axes.append(ax)
            if gallery:
                counts=s['nominal_measured'][s['plan']['reachable'].astype(bool)].sum(0);table=s['normals'][:,2]<=-np.cos(np.deg2rad(10));visible=(counts>0)&~table
                scale=old['color_scale'];artist=ax.scatter(*s['points'][visible].T,c=counts[visible],cmap=theme.CMAP,vmin=scale['vmin'],vmax=scale['vmax'],s=.2,rasterized=True)
                ax.scatter(*s['points'][table].T,color=theme.role('table_facing'),s=.1,alpha=.2,rasterized=True)
                unseen=(counts==0)&~table;ax.scatter(*s['points'][unseen].T,color=theme.role('unseen'),marker='x',s=.1,alpha=.2,rasterized=True)
                ax.set_axis_off();a=s['attribution']
                credit=f'{s["part"]}: '+textwrap.fill(a['name'],35)+'\n'+a['author']+'\n'+('CC BY-SA' if a['share_alike'] else 'CC BY')+'; version unknown\n'+a['thing_url']
                ax.set_title(credit,fontsize=10)
            else:
                p=s['plan'];az=np.degrees(np.arctan2(p['direction'][:,1],p['direction'][:,0]));el=np.degrees(np.arcsin(np.clip(p['direction'][:,2],-1,1)))
                for mask,role,marker in [(p['reachable'],'reachable','o'),(~p['ik_reachable'],'ik_unreachable','x'),(p['ik_reachable']&~p['reachable'],'collision_masked','^')]:
                    ax.scatter(az[mask],el[mask],color=theme.role(role),marker=marker,label=role.replace('_',' '))
                ax.set(title=str(s['part']),xlabel='Azimuth (deg)',ylabel='Elevation (deg)',xlim=(-180,180),ylim=(0,90));ax.legend(fontsize=8)
        if gallery:fig.colorbar(artist,ax=axes,shrink=.4,label='Reachable measuring passes (actual positive range)')
        title=split.upper()+(': nominal surface accessibility' if gallery else ': saved candidate masks')
        caption='Thingi10K; 200 mm scaling, area-weighted samples and derived visibility. Source license versions unknown. Table-facing: normal within 10° of downward; other unseen samples separate.' if gallery else 'Saved nominal geometry and reachability; no new IK or visibility. Part frame, z upward; angle axes in degrees.'
    decorate(fig,axes,title,caption,lang,slide)
    for ax,labels in named_axes:
        ax.set_yticklabels([textwrap.fill(label,18) if slide else label for label in labels])
        for label in ax.get_yticklabels():label.set_linespacing(.85)
    return fig,title,caption,styles

REPORTS={'p004_a_quality_latency':'a','p004_b_learning_runs':'b','p004_c_test_distributions':'c','p004_c2_paired_forward':'c2',
 'p004_d_val_coverage_time':'d','p004_e_val_time_components':'e','p004_e_test_motion_time':'motion',
 'p004_f_val_gallery':'f_val','p004_f_train_gallery':'f_train','p004_g_val_candidate_sphere':'g_val','p004_g_train_candidate_sphere':'g_train'}
SLIDES={'p004b_a_quality_latency':'a','p004b_b_learning':'b','p004b_c_test_parts':'c','p004b_c2_paired_forward':'c2',
        'p004b_e_val_time_components':'e','p004b_collision_rates':'collision'}

def generate(*,slides=False,figures=None,output_dir=None,version='v2'):
    from nbv.evaluation import git_state
    if git_state()['dirty']:raise ValueError('Commit rendering code before publishing versioned figures')
    if not re.fullmatch(r'v[1-9][0-9]*',version):raise ValueError('Expected a version token such as v3')
    root=Path('results/figures/slides' if slides else 'results/figures');catalog=SLIDES if slides else REPORTS
    selected=list(catalog) if figures is None else list(figures)
    if not selected or len(set(selected))!=len(selected) or set(selected)-set(catalog):raise ValueError('Select distinct known figure names')
    destination=Path(output_dir) if output_dir is not None else root
    variants=('a','b') if slides else (None,)
    names=[stem+'_'+version+('_'+v if v else '') for stem in selected for v in variants]
    manifest=destination/('manifest_'+version+'.json')
    targets=[destination/(name+ext) for name in names for ext in ('.png','.pdf','.svg','.json')]+[manifest]
    if any(p.exists() for p in targets):raise FileExistsError('Versioned export destination already contains a target')
    workspace=prepare_export_workspace();files={};outputs=[]
    with tempfile.TemporaryDirectory(prefix='theme-'+version+'-',dir=workspace) as directory:
        stage=Path(directory).resolve();assert stage.is_relative_to(workspace)
        for stem in selected:
            old=json.loads((root/(stem+'.json')).read_text(encoding='utf-8'))
            for variant in variants:
                fig,title,caption,styles=draw(catalog[stem],old,'tr' if slides else 'en',variant)
                name=stem+'_'+version+('_'+variant if variant else '')
                save_figure(fig,stage/name,old,language='tr' if slides else 'en',slide=slides,render_styles=styles,title=title,caption=caption)
                outputs.append(name)
                for ext in ('.png','.pdf','.svg','.json'):
                    p=stage/(name+ext);files[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
                print('Validated '+version+' figure:',name,flush=True)
        # Publish only fully validated new artifacts, and clean staging on success or failure.
        destination.mkdir(parents=True,exist_ok=True)
        for name in files:shutil.copyfile(stage/name,destination/name)
        source=root/'manifest_v1.json' if slides else Path('results/visualization/static_figures_v1.json')
        reference=Path(os.path.relpath(source,destination)).as_posix()
        with manifest.open('x',encoding='utf-8',newline='\n') as stream:
            json.dump(dict(theme_version='v2',figure_revision=version,files=files,figures=outputs,source_manifest=reference),stream,indent=2)
    return outputs
