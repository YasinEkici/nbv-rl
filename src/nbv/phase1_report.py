"""Registry-driven Phase 1 reports: predictions before opening, verdicts after opening."""
import csv
import json
from pathlib import Path
import numpy as np
from nbv import evaluation, hypotheses
from nbv.env import ENV_VERSION
from nbv.frozen_evaluation import BASELINES
from nbv.report_common import model_limitations

ARMS=('sarsa/linear','regq/linear','myopic/linear','sarsa/none','sarsa/convex','sarsa/value_estimate')


def curve_diagnosis(curves,tuning_score):
    checkpoints={}
    for label,c in curves.items():
        checkpoints[label]={step:{key:c[key][c['steps'].index(step)] for key in ('mean','low','high')}
                            for step in (200000,1000000,2000000)}
    auc={p:float(np.trapezoid(curves[f'sarsa/{p}/11']['mean'],curves[f'sarsa/{p}/11']['steps']))
         for p in ('linear','none','convex','value_estimate')}
    start=checkpoints['sarsa/linear/11'][200000]['mean']
    end=checkpoints['sarsa/linear/11'][2000000]['mean']
    return {'checkpoints':checkpoints,'auc':auc,'auc_relative_range':(max(auc.values())-min(auc.values()))/min(auc.values()),
            'tuning_score':tuning_score,'official_200k':start,'official_2m':end,
            'dominant':'oscillation' if abs(end-start)>abs(start-tuning_score) else 'selection_optimism'}


def rows(path='results/registry.csv'):
    return list(csv.DictReader(open(path,encoding='utf-8')))


def record_groups(registry,split,kind='tail',outputs='outputs'):
    groups={}
    for row in registry:
        if row['status']!='canonical' or row['env_version']!=ENV_VERSION or row['part_split']!=split:
            continue
        extra=json.loads(row['extra_json'])
        baseline=(row['plan']=='001' and split in ('train','val') and row['method'] in BASELINES)
        baseline|=(extra.get('tag')=='p002test' and extra.get('baseline',False))
        learner=extra.get('tag')==('p002train' if split=='train' else 'p002test') and extra.get('weight_kind')==kind
        if not (baseline or learner) or extra.get('n_candidates')!=48: continue
        label=row['method'] if baseline else f"{row['method']}/{extra['phi']}"
        if 'episode_metrics' in extra:
            records=extra['episode_metrics']
        else:
            records=[json.loads(line) for line in (Path(outputs)/row['run_id']/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
        records=[e|{'success_without_repair':e.get('success_without_repair',e.get('planner',{}).get('success_without_repair'))}
                 for e in records]
        groups.setdefault(label,[]).extend(records)
    return groups


def nested(records,key):
    return evaluation.nest([e for e in records if e.get(key) is not None],key)


def mean(records,key):
    data=nested(records,key)
    return evaluation.nested_mean(data) if data else None


def interval(records,key,n_boot=10000):
    vals=[float(e[key]) for e in records if e.get(key) is not None]
    if not vals: return None
    if min(vals)==max(vals): return {'mean':vals[0],'ci_low':vals[0],'ci_high':vals[0]}
    return evaluation.hierarchical_bootstrap(nested(records,key),n_boot=n_boot,seed=0)


def fmt(value):
    if value is None: return 'n/a'
    if isinstance(value,dict): return f"{value['mean']:.6f} [{value['ci_low']:.6f}, {value['ci_high']:.6f}]"
    return f'{value:.6f}'


def pretest_report(registry='results/registry.csv',out='results/pretest_v1.md',predictions='results/pretest_predictions_v1.json'):
    records=rows(registry)
    selected=json.loads(Path('results/selected_settings_v1.json').read_text())
    curves=json.loads(Path('results/val_curves_v1.json').read_text())
    d=curve_diagnosis(curves,selected['sarsa']['mean'])
    # Tail-policy final VAL values from registered histories, not from rounded prose.
    scores={label:[] for label in ARMS}
    for r in records:
        e=json.loads(r['extra_json'])
        if r['status']=='canonical' and e.get('tag')=='p002' and e.get('feature_count')==11:
            scores[f"{r['method']}/{e['phi']}"].append(e['history'][-1]['normalized_cycle_time'])
    if any(len(v)!=10 for v in scores.values()): raise ValueError('complete official VAL arms required')
    final={key:float(np.mean(value)) for key,value in scores.items()}
    pred={'H1':'rejected','H-forward':'rejected','H-shaping':'rejected' if d['auc_relative_range']<.1 else 'borderline'}
    data=d|{'final_val':final,'predictions':pred,'main_method':'sarsa','no_selection_change':True}
    Path(predictions).write_text(json.dumps(data,indent=2),encoding='utf-8')
    lines=['# Pretest curve diagnosis and predictions (written before test opening)','',
           '| Arm | 200000 steps: mean [seed min, max] | 1000000 steps | 2000000 steps |', '|---|---|---|---|']
    for label,points in d['checkpoints'].items():
        cells=[f"{points[s]['mean']:.6f} [{points[s]['low']:.6f}, {points[s]['high']:.6f}]" for s in (200000,1000000,2000000)]
        lines.append('| '+label+' | '+' | '.join(cells)+' |')
    lines+=['',f"SARSA selected tuning score: {d['tuning_score']:.6f}; official 200000-step mean: {d['official_200k']:.6f}; official 2000000-step mean: {d['official_2m']:.6f}.",
            f"Dominant descriptive explanation under the predeclared comparison: {d['dominant']}. Selection optimism and bounded policy oscillation can coexist; this is not causal identification or a theorem guarantee.",
            'The tuning score averages two tail evaluations; the 200000-step official point uses current weights before its official tail window. Seeds are disjoint and the weight rules differ at those points, limiting direct attribution.',
            'Last-weight versus tail-policy differences support an oscillation interpretation; RegQ is empirically steadier. No extra training or selection.', '',
            '| Phi | VAL learning-curve area (step × normalized time) | Final VAL minus linear |', '|---|---:|---:|']
    for p,a in d['auc'].items(): lines.append(f"| {p} | {a:.6f} | {final['sarsa/'+p]-final['sarsa/linear']:.6f} |")
    lines += ['',f"Relative AUC range (worst−best)/best: {d['auc_relative_range']:.6%}. Trapezoid on observed 100000–2000000 steps (D-055).",
              'Predictions: H1′ rejected; H-forward rejected; H-shaping '+pred['H-shaping']+'. Main method remains true online SARSA(lambda).',
              'The pre-tuning small-gap prediction agrees with official VAL; the post-tuning reversal does not. This illustrates selection optimism, without proving it is the sole cause.',
              'RegQ is the best VAL learner and the only learner beating greedy-IG/cost; it remains descriptive, with no hypothesis or primary-method promotion.']
    Path(out).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return data


def table(groups,per_part=False,n_boot=None):
    lines=['| Arm | Part | Normalized cycle time | Raw seconds | Success |','|---|---|---:|---:|---:|']
    for label in [*ARMS,*BASELINES]:
        if label not in groups: raise ValueError(f'missing report arm {label}')
        records=groups[label]
        parts=sorted({str(e['part']) for e in records}) if per_part else ['split']
        for part in parts:
            subset=[e for e in records if str(e['part'])==part] if per_part else records
            values=[interval(subset,k,n_boot) if n_boot is not None else mean(subset,k)
                    for k in ('normalized_cycle_time','cycle_time','success')]
            lines.append('| '+label+' | '+part+' | '+' | '.join(fmt(x) for x in values)+' |')
    return '\n'.join(lines)


def train_report(registry='results/registry.csv',out='results/train_v1.md'):
    source=rows(registry)
    primary=record_groups(source,'train')
    secondary=record_groups(source,'train','final')
    text='# Frozen-policy train evaluation\n\nNo training or selection. All 60 canonical checkpoints, both weight kinds; D-041 train baselines reused.\n\n'
    text+=table(primary,True)+'\n\nSplit means (equal part/variant/seed weights):\n\n'+table(primary)
    text+='\n\nSecondary final weights:\n\n'+table(secondary)
    Path(out).write_text(text+model_limitations(),encoding='utf-8')


def test_report(registry='results/registry.csv',out='results/test_v1.md',n_boot=10000,
                predictions_path='results/pretest_predictions_v1.json',curves_path='results/val_curves_v1.json',
                incident_path='outputs/p002-test-incident-1.json',resume_path='outputs/p002-test-resumed-1.json'):
    source=rows(registry)
    primary=record_groups(source,'test')
    secondary=record_groups(source,'test','final')
    training=record_groups(source,'train')
    predictions=json.loads(Path(predictions_path).read_text())
    def data(label): return nested(primary[label],'normalized_cycle_time')
    h1=hypotheses.h1(data('sarsa/linear'),data('greedy_cost'))
    forward=hypotheses.forward(data('sarsa/linear'),data('myopic/linear'))
    curves=json.loads(Path(curves_path).read_text())
    phis=('linear','none','convex','value_estimate')
    shaping=hypotheses.shaping({p:list(zip(curves[f'sarsa/{p}/11']['steps'],curves[f'sarsa/{p}/11']['mean'])) for p in phis},
                               {p:data('sarsa/'+p) for p in phis})
    verdicts={'H1':h1,'H-forward':forward,'H-shaping':shaping}
    text='# Single-opening test evaluation\n\n'
    if Path(incident_path).is_file() and Path(resume_path).is_file():
        incident=json.loads(Path(incident_path).read_text(encoding='utf-8'))
        resumed=json.loads(Path(resume_path).read_text(encoding='utf-8'))
        text+=(f"Incident 1: the opening at {incident['opening_commit']} stopped while hashing a missing nominal point cache, before environment construction or test performance. D-062 records the evidence and the user's authorization for exactly one continuation. The original claim was preserved; nominal-only helpers/plans and committed SHA-256 manifests were verified before the exclusive continuation at {resumed['commit']}. No settings, methods, hypotheses or primary-method choice changed.\n\n")
    text+='Primary: final-10% tail policies. Equal part/variant/seed weights; censored times retained with success. Exploratory 16-feature arm excluded.\n'
    text+='\nPer part: mean [bootstrap interval over variants, then seeds].\n\n'+table(primary,True,n_boot)
    text+='\n\nSplit summary: mean [hierarchical bootstrap interval over parts, variants, seeds]. No significance tests.\n\n'+table(primary,False,n_boot)
    text+='\n\nHypotheses and pretest predictions:\n\n| Hypothesis | Pretest prediction | Test verdict | Prediction comparison |\n|---|---|---|---|\n'
    for name,result in verdicts.items():
        verdict='supported' if result['supported'] else 'rejected'
        prediction=predictions['predictions'][name]
        comparison='consistent' if verdict==prediction else ('borderline prediction resolved' if prediction=='borderline' else 'inconsistent')
        text+=f'| {name} | {prediction} | {verdict} | {comparison} |\n'
    text+='\nH1′ details:\n\n| Part | SARSA | Threshold | Meets | Fraction of greedy–ILP gap closed | SARSA minus MCTS |\n|---|---:|---:|---|---:|---:|\n'
    a=evaluation.per_part(data('sarsa/linear')); m=evaluation.per_part(data('mcts'))
    for p in a:
        gap=h1['gap_closed'][p]
        text+=f"| {p} | {a[p]:.6f} | {h1['thresholds'][p]:.6f} | {h1['parts_meeting'][p]} | {fmt(gap)} | {a[p]-m[p]:.6f} |\n"
    text+=f"\nH-forward mean paired SARSA−myopic difference: {forward['mean_difference']:.6f}; per part: "+json.dumps(forward['paired_difference'],sort_keys=True)+'.\n'
    text+='If rejected, the predeclared interpretation is that myopic information captures the added contribution of bootstrapping under this representation; rejection is not proof of policy equivalence.\n'
    text+=f"\nH-shaping VAL relative AUC range: {shaping['relative_auc_range']:.6%}; speed criterion: {shaping['speed_changes']}; final-test closeness criterion: {shaping['final_close']}.\n"
    text+='\n| Phi | VAL learning-curve area | Test normalized time minus linear |\n|---|---:|---:|\n'
    for p in phis: text+=f"| {p} | {shaping['auc'][p]:.6f} | {shaping['final_difference'][p]:.6f} |\n"
    text+='\nPBRS preserves the objective/optimal policy under its assumptions, not learning speed. None versus linear compares initial values (D-038/Wiewiora).\n'
    text+='\nSecondary metrics (tail policy; passes to 95% conditional on success):\n\n| Arm | Passes | Coverage@5 | Coverage@10 | Coverage AUC | Joint motion rad | Repair passes | Train−test normalized time | Train−test raw seconds | Train−test success |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for label in [*ARMS,*BASELINES]:
        records=primary[label]
        values=[mean(records,k) for k in ('passes_to_95','coverage_at_5','coverage_at_10','auc_coverage_time','total_joint_motion','repair_passes')]
        values += [mean(training[label],'normalized_cycle_time')-mean(records,'normalized_cycle_time'),
                   mean(training[label],'cycle_time')-mean(records,'cycle_time'),
                   mean(training[label],'success')-mean(records,'success')]
        text+='| '+label+' | '+' | '.join(fmt(x) for x in values)+' |\n'
    text+='\nSecondary coverage references and open-loop repair:\n\n| Arm | NSC@5 | NSC@10 | Raw coverage@5 | Raw coverage@10 | Success without repair |\n|---|---:|---:|---:|---:|---:|\n'
    for label in [*ARMS,*BASELINES]:
        values=[mean(primary[label],k) for k in ('nsc_at_5','nsc_at_10','raw_at_5','raw_at_10','success_without_repair')]
        text+='| '+label+' | '+' | '.join(fmt(x) for x in values)+' |\n'
    regq=[json.loads(r['extra_json']) for r in source if json.loads(r['extra_json']).get('tag')=='p002' and r['method']=='regq']
    text+=f"\nRegQ training divergence: {sum(e['diverged'] for e in regq)}/{len(regq)}; descriptive only. Primary method remains true online SARSA(lambda). Alpha-scale asymmetry and Phi-setting inheritance remain limitations.\n"
    text+='\nAppendix: final weights (secondary, never selected).\n\n'+table(secondary,True,n_boot)+'\n\n'+table(secondary,False,n_boot)
    Path(out).write_text(text+model_limitations(),encoding='utf-8')
    Path(out).with_suffix('.json').write_text(json.dumps(verdicts,indent=2),encoding='utf-8')
    return verdicts
