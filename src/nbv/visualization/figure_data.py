"""Registry-only figure inputs. No part cache, episode output or environment imports."""
import csv
import json
from nbv import evaluation


def rank_runs(runs):
    if len(runs)!=10 or {r['seed'] for r in runs}!=set(range(10)): raise ValueError('ten original seeds required')
    ordered=sorted(runs,key=lambda r:(r['score'],r['seed']))
    alternative=sorted(runs,key=lambda r:(sum(r['curve'])/len(r['curve']),r['seed']))
    return dict(ordered=[r['seed'] for r in ordered],best=ordered[0]['seed'],median=ordered[4]['seed'],worst=ordered[-1]['seed'],
        alternative_curve_mean_order=[r['seed'] for r in alternative],
        rule='Official tail-policy VAL score ascending, then seed ascending; lower median = sorted index 4 (fifth actual run). Not percentiles.',
        adaptation='Ten seeds, not paper 100 runs; official-policy final VAL score, not whole-run mean return. Alternative 20-point VAL-curve-mean order is information only.')


class RegistryFigures:
    def __init__(self,path='results/registry.csv'):
        self.path=str(path)
        with open(path,encoding='utf-8') as stream: self.rows=list(csv.DictReader(stream))

    def test_records(self):
        records=[]
        for row in self.rows:
            if row['status']!='canonical' or row['part_split']!='test': continue
            extra=json.loads(row['extra_json'])
            if extra.get('tag')!='p002test' or not(extra.get('baseline') or extra.get('weight_kind')=='tail'): continue
            label=row['method'] if extra.get('baseline') else f"{row['method']}/{extra['phi']}"
            records.extend(e|{'label':label,'source_run':row['run_id'],'source_commit':row['commit']} for e in extra['episode_metrics'])
        return records

    def paired(self):
        records=self.test_records()
        a=evaluation.nest([e for e in records if e['label']=='sarsa/linear'],'normalized_cycle_time')
        b=evaluation.nest([e for e in records if e['label']=='myopic/linear'],'normalized_cycle_time')
        difference=evaluation.paired_difference(a,b)
        return {'variants':difference,'mean':evaluation.nested_mean(difference),
                'source_runs':sorted({e['source_run'] for e in records if e['label'] in ('sarsa/linear','myopic/linear')})}

    def learning(self,label):
        result=[]
        for row in self.rows:
            extra=json.loads(row['extra_json'])
            if extra.get('tag')=='p002' and row['status']=='canonical' and f"{row['method']}/{extra['phi']}"==label:
                history=extra['history']
                if len(history)!=20: raise ValueError('original 20-point history required')
                result.append(dict(seed=int(row['seed']),source_run=row['run_id'],source_commit=row['commit'],
                    steps=[h['step'] for h in history],curve=[h['normalized_cycle_time'] for h in history],
                    score=history[-1]['normalized_cycle_time']))
        return result

    def timing(self):
        matches=[r for r in self.rows if json.loads(r['extra_json']).get('tag')=='p002planning']
        if len(matches)!=1: raise ValueError('one original timing record required')
        row=matches[0]
        return json.loads(row['extra_json'])|{'source_run':row['run_id'],'source_commit':row['commit']}
