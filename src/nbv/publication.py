"""Registry-derived Phase 1 closure tables and presentation figures; never loads part caches."""
import csv
import json
import os
import re
from pathlib import Path
import numpy as np
from nbv.phase1_report import rows,record_groups,mean,ARMS,BASELINES
from nbv.report_common import model_limitations


def reference_table(keys,literature='docs/LITERATURE.md'):
    sections={}
    for section in Path(literature).read_text(encoding='utf-8').split('\n## '):
        heading=section.split('\n',1)[0].removeprefix('## ')
        if ' — ' in heading: sections[heading.split(' — ',1)[0]]=section
    lines=['| Key | Citation | Status | Source/notes |','|---|---|---|---|']
    for key in keys:
        section=sections[key]; title=section.split('\n',1)[0].removeprefix('## ').split(' — ',1)[1]
        status=re.search(r'^Status: (.+)$',section,re.M).group(1)
        if not status.startswith(('verified-source','secondary','verified (')): raise ValueError(f'unverified reference: {key}')
        urls=re.findall(r'https?://[^\s;]+',section)
        source=f'[source]({urls[0].rstrip(".,")})' if urls else '[literature notes](docs/LITERATURE.md)'
        lines.append(f'| {key} | {title} | {status} | {source} |')
    return '\n'.join(lines)


def replace_block(path,name,content):
    path=Path(path)
    if not path.exists(): return
    text=path.read_text(encoding='utf-8')
    start,end=f'<!-- BEGIN GENERATED: {name} -->',f'<!-- END GENERATED: {name} -->'
    if start in text:
        text=re.sub(re.escape(start)+'.*?'+re.escape(end),lambda _:start+'\n'+content.strip()+'\n'+end,text,flags=re.S)
        path.write_text(text,encoding='utf-8')


def report(registry='results/registry.csv'):
    source=rows(registry)
    train=record_groups(source,'train'); test=record_groups(source,'test'); final=record_groups(source,'test','final')
    val=record_groups(source,'val')
    for row in source:
        extra=json.loads(row['extra_json'])
        if row['status']=='canonical' and extra.get('tag')=='p002':
            label=f"{row['method']}/{extra['phi']}"
            val.setdefault(label,[]).extend(json.loads(line) for line in (Path('outputs')/row['run_id']/'episodes.jsonl').read_text(encoding='utf-8').splitlines())
    splits={'train':train,'val':val,'test':test}
    finding=['| SARSA Phi | Train | VAL | Test |','|---|---:|---:|---:|']
    for phi in ('linear','none','convex','value_estimate'):
        finding.append('| '+phi+' | '+' | '.join(f"{mean(g['sarsa/'+phi],'normalized_cycle_time'):.6f}" for g in splits.values())+' |')
    finding.append('| MCTS reference | '+' | '.join(f"{mean(g['mcts'],'normalized_cycle_time'):.6f}" for g in splits.values())+' |')
    finding='\n'.join(finding)
    verdict=json.loads(Path('results/test_v1.json').read_text(encoding='utf-8'))
    delta=verdict['H-forward']['mean_difference']
    finding+=f'\n\nH-forward paired SARSA-minus-myopic difference: {delta:.15f}; missing improvement to the predeclared 0.02 threshold: {.02+delta:.15f}. No rounding-based reinterpretation.'
    Path('results/phase1_findings_v1.md').write_text('# Descriptive cross-split finding\n\n'+finding+'\n\nvalue_estimate is the lowest SARSA mean on each split. This is descriptive, outside the hypotheses; no arm or setting is selected from test results. Cross-split consistency is supporting evidence, not an independent replication. Six tested arms expose a test-only best-arm claim to selection optimism.\n',encoding='utf-8')
    for path in ('docs/plans/002-classical-rl.md','docs/plans/done/002-classical-rl.md','docs/DECISIONS.md'):
        replace_block(path,'cross_split',finding)
    timing=[json.loads(r['extra_json']) for r in source if json.loads(r['extra_json']).get('tag')=='p002planning']
    planning='Timing has not yet been measured.'
    if timing:
        if len(timing)!=1: raise ValueError('one timing record required')
        t=timing[0]
        planning='| Policy | Source split | Mean ms/decision | Median ms | P95 ms | Calls |\n|---|---|---:|---:|---:|---:|\n'
        for label,result in t['timings'].items():
            planning+=f"| {label} | {t['split']} | {result['planning_time_ms_per_decision']:.6f} | {result['median_ms']:.6f} | {result['p95_ms']:.6f} | {result['decisions']} |\n"
        planning+='\n'+t['definition']+' Source checkpoints are fixed official seed 0; timings characterize this machine and state sample, not a seed-level quality comparison.\n'
        Path('results/planning_time_v1.md').write_text('# Deployed decision latency (exploratory)\n\n'+planning,encoding='utf-8')
    ceiling='| Split | Mean reachable raw-surface share | Range over base-part means |\n|---|---:|---:|\n'
    for split,group in splits.items():
        records=[e|{'share':e['raw_at_5']/e['coverage_at_5']} for e in group['greedy_cost'] if e['coverage_at_5']>0]
        per={p:mean([e for e in records if str(e['part'])==p],'share') for p in {str(e['part']) for e in records}}
        ceiling+=f'| {split} | {np.mean(list(per.values())):.2%} | {min(per.values()):.2%}–{max(per.values()):.2%} |\n'
    Path('results/raw_surface_ceiling_v1.md').write_text('# Reachable raw-surface ceiling\n\n'+ceiling+'\nDerived from registered raw@5 / primary@5 for the same variant; this recovers reachable-union / all samples. No test cache access.\n',encoding='utf-8')
    selected=[r for r in source if r['plan']=='002']
    with Path('results/plan002_registry_v1.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['run_id','tag','status','part_split','method','seed','budget_steps','commit'])
        writer.writeheader()
        writer.writerows({k:(json.loads(r['extra_json']).get('tag','feature_check') if k=='tag' else r[k]) for k in writer.fieldnames} for r in selected)
    test_text=Path('results/test_v1.md').read_text(encoding='utf-8')
    blocks={
        'test_primary':test_text.split('Per part: mean [bootstrap interval over variants, then seeds].\n\n',1)[1].split('\n\nSplit summary:',1)[0],
        'test_summary':test_text.split('No significance tests.\n\n',1)[1].split('\n\nHypotheses',1)[0],
        'hypotheses':test_text.split('Hypotheses and pretest predictions:\n\n',1)[1].split('\nH1',1)[0],
        'cross_split':finding,'planning':planning,'raw_ceiling':ceiling,
        'secondary':test_text.split('Secondary metrics (tail policy; passes to 95% conditional on success):\n\n',1)[1].split('\n\nSecondary coverage',1)[0],
        'model_limitations':model_limitations(collision_prefix='results/').strip().replace('## Model limitations','### Model limitations',1),
    }
    reference_keys=['thingi10k','vanseijen2016','regq','ng1999','grzes2017','wiewiora2003','canonaco2024',
        'patterson2024','smith_winkler2006','eimer2023','zhang2023_sarsa','saravanan2020',
        'landgraf2021','kaiser2025_thesis','koch2025','roos2025','cheng2021_hurl','wang2016_dueling','silver2018_rpl','johannink2019']
    blocks['references']=reference_table(reference_keys)
    for name,content in blocks.items(): replace_block('README.md',name,content)
    os.environ.setdefault('MPLCONFIGDIR',str(Path('scratch/matplotlib').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    destination=Path('results/figures'); destination.mkdir(exist_ok=True)
    def save(fig,name):
        for ext in ('png','pdf'):
            metadata={'Creator':'NBV-RL registry report'}
            if ext=='pdf': metadata|={'CreationDate':None,'ModDate':None}
            fig.savefig(destination/f'{name}.{ext}',dpi=180,metadata=metadata)
        plt.close(fig)
    labels=[*ARMS,*BASELINES]
    fig,axes=plt.subplots(1,3,figsize=(18,7),sharey=True,sharex=True,layout='constrained')
    for ax,part in zip(axes,sorted({str(e['part']) for e in test['sarsa/linear']})):
        values=[mean([e for e in test[label] if str(e['part'])==part],'normalized_cycle_time') for label in labels]
        ax.barh(labels,values,color=['#2166ac' if '/' in label else '#888888' for label in labels])
        ax.axvline(1,color='black',lw=.8); ax.set(title=f'Test part {part}',xlabel='Normalized cycle time (lower is better)')
    axes[0].invert_yaxis(); fig.suptitle('Tail policies; random censored times included (success rates in test_v1.md)')
    save(fig,'phase1_test_by_part_v1')
    curves=json.loads(Path('results/val_curves_v1.json').read_text(encoding='utf-8'))
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),sharey=True,layout='constrained')
    for ax,label in zip(axes,('sarsa/linear','regq/linear','sarsa/value_estimate')):
        c=curves[label+'/11']; ax.plot(c['steps'],c['mean']); ax.fill_between(c['steps'],c['low'],c['high'],alpha=.2)
        ax.set(title=label,xlabel='Training environment steps')
    axes[0].set_ylabel('VAL normalized cycle time'); fig.suptitle('Official seed means and seed min–max; identical axes')
    save(fig,'phase1_three_val_curves_v1')
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained'); x=np.arange(len(ARMS))
    ax.bar(x-.2,[mean(test[label],'normalized_cycle_time') for label in ARMS],.4,label='Tail mean')
    ax.bar(x+.2,[mean(final[label],'normalized_cycle_time') for label in ARMS],.4,label='Final weights (secondary)')
    ax.set(xticks=x,xticklabels=ARMS,ylabel='Test normalized cycle time',title='Fixed tail versus final policies; success reported separately')
    ax.tick_params(axis='x',rotation=15); ax.legend(loc='upper left',bbox_to_anchor=(1.01,1)); save(fig,'phase1_tail_vs_final_v1')
    return blocks
