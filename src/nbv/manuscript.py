"""Source-derived public manuscript; never evaluates policies or resamples results."""
import csv
import hashlib
import json
from pathlib import Path
import re
from statistics import fmean

import yaml


ROOT = Path(__file__).resolve().parents[2]
LABELS = {
    'sarsa/linear': 'SARSA · doğrusal potansiyel (ana yöntem)',
    'sarsa/none': 'SARSA · şekillendirme yok',
    'sarsa/convex': 'SARSA · dışbükey potansiyel',
    'sarsa/value_estimate': 'SARSA · değer tahmini',
    'regq/linear': 'RegQ', 'myopic/linear': 'Miyop öğrenici',
    'greedy_cost': 'Greedy-IG/maliyet', 'greedy_ig': 'Greedy-IG',
    'mcts': 'MCTS', 'ilp_tsp': 'ILP + TSP', 'joint_opt': 'Ortak planlayıcı',
    'random': 'Rastgele', 'fixed': 'Sabit desen',
}


def _load(root, name):
    return json.loads((root/name).read_text(encoding='utf-8'))


def _mean(records, field):
    parts = {}
    for row in records:
        parts.setdefault(str(row['part']), {}).setdefault(row['variant'], []).append(float(row[field]))
    return fmean(fmean(fmean(seeds) for seeds in variants.values()) for variants in parts.values())


def _test_records(rows, kind='tail'):
    groups = {}
    for row in rows:
        extra = json.loads(row['extra_json'])
        if row['status'] != 'canonical' or row['part_split'] != 'test' or extra.get('tag') != 'p002test':
            continue
        if extra.get('baseline'):
            label = row['method']
        elif extra.get('weight_kind') == kind:
            label = row['method']+'/'+extra['phi']
        else:
            continue
        groups.setdefault(label, []).extend(extra['episode_metrics'])
    return groups


def render(root=ROOT):
    """Return manuscript text and a source/formula record for every rendered claim."""
    root = Path(root)
    template = (root/'assets/report/readme_tr.md.in').read_text(encoding='utf-8')
    with (root/'results/registry.csv').open(encoding='utf-8') as stream:
        registry = list(csv.DictReader(stream))
    extras = [(row, json.loads(row['extra_json'])) for row in registry]
    official = [(row, e) for row, e in extras if row['status']=='canonical' and e.get('tag')=='p002']
    groups = _test_records(registry)
    verdict = _load(root, 'results/test_v1.json')
    selected = _load(root, 'results/selected_settings_v1.json')
    publication = _load(root, 'provenance/publication.json')
    claims = []
    values = {}

    def add(key, value, source, source_key, *, places=None, derived=False, formula=None):
        text = str(value) if places is None else f'{value:.{places}f}'
        if isinstance(value, float) or places is not None:
            text = text.replace('.', ',')
        claims.append(dict(key=key, raw=value, text=text, source=source, source_key=source_key,
                           derived=derived, formula=formula))
        values[key] = text
        return text

    for label, records in groups.items():
        add('test.'+label, _mean(records,'normalized_cycle_time'), 'results/registry.csv',
            f'canonical test tail/baseline: {label}', places=6, derived=True,
            formula='mean(parts -> mean(variants -> mean(seeds)))')

    split = yaml.safe_load((root/'configs/split_v1.yaml').read_text(encoding='utf-8'))
    protocol=(root/'docs/PROTOCOL.md').read_text(encoding='utf-8')
    add('assignment',int(re.search(r'Homework (\d+)',protocol).group(1)),'docs/PROTOCOL.md','assignment')
    env = yaml.safe_load((root/'configs/env_v4.yaml').read_text(encoding='utf-8'))
    robot = yaml.safe_load((root/'configs/robot_v2.yaml').read_text(encoding='utf-8'))
    sensor = yaml.safe_load((root/'configs/sensor_v1.yaml').read_text(encoding='utf-8'))
    for key, value in split['counts'].items():
        add('parts.'+key, value, 'configs/split_v1.yaml', 'counts.'+key)
    add('parts.total', len(split['parts']), 'configs/split_v1.yaml', 'parts', derived=True, formula='len(parts)')
    add('parts.variants', split['variants_per_part'], 'configs/split_v1.yaml','variants_per_part')
    add('target', env['coverage_target']*100, 'configs/env_v4.yaml', 'coverage_target', places=0, derived=True, formula='100 * coverage_target')
    add('bonus', env['reward']['success_bonus_s'], 'configs/env_v4.yaml','reward.success_bonus_s',places=0)
    add('passes', next(e['n_candidates'] for _,e in official),'results/registry.csv','official.n_candidates')
    add('pass_limit',env['k_max'][48],'configs/env_v4.yaml','k_max[48]')
    for key, value, source_key in [('move_speed',robot['motion']['v_rad_s'],'motion.v_rad_s'),('move_accel',robot['motion']['a_rad_s2'],'motion.a_rad_s2'),('tcp',robot['tcp_offset_mm'],'tcp_offset_mm'),('ik_samples',robot['ik']['n_samples'],'ik.n_samples')]:
        add(key,value,'configs/robot_v2.yaml',source_key)
    add('scan_speed',sensor['scan_speed_mm_s'],'configs/sensor_v1.yaml','scan_speed_mm_s',places=0)
    add('working_distance',sensor['working_distance_mm'],'configs/sensor_v1.yaml','working_distance_mm',places=0)
    add('line_width',sensor['line_width_mm'],'configs/sensor_v1.yaml','line_width_mm',places=0)
    add('triangulation',sensor['triangulation_angle_deg'],'configs/sensor_v1.yaml','triangulation_angle_deg',places=0)
    add('feature_count',next(e['feature_count'] for _,e in official),'results/registry.csv','official.feature_count')
    add('official_steps',int(official[0][0]['budget_steps']),'results/registry.csv','official.budget_steps')
    add('official_seeds',len({row['seed'] for row,_ in official}),'results/registry.csv','official.seed',derived=True,formula='distinct seed count')
    add('official_arms',len({(row['method'],e['phi']) for row,e in official}),'results/registry.csv','official method/Phi',derived=True,formula='distinct arm count')
    timing=next(e for _,e in extras if e.get('tag')=='p002planning')
    batch=next(e for _,e in extras if e.get('tag')=='p002timing')
    add('training_hours',batch['elapsed_s']/3600,'results/registry.csv','batch elapsed_s',places=3,derived=True,formula='elapsed_s / 3600')
    add('workers',batch['workers'],'results/registry.csv','batch workers')
    add('hardware',publication['hardware'],'provenance/publication.json','hardware')
    add('python', (root/'.python-version').read_text(encoding='utf-8').strip(),'.python-version','version')
    add('weights_sha256',publication['weights']['sha256'],'provenance/publication.json','weights.sha256')
    add('weights_bytes',publication['weights']['bytes'],'provenance/publication.json','weights.bytes')
    for key,value in publication['learning'].items():
        add(key,value,'provenance/publication.json','learning.'+key)
    for key,value in publication['thresholds'].items():
        add('threshold.'+key,value,'provenance/publication.json','thresholds.'+key)
    add('threshold.shaping_percent',100*publication['thresholds']['shaping_speed'],'provenance/publication.json','thresholds.shaping_speed',places=0,derived=True,formula='100 * speed threshold')
    from nbv.evaluation import hierarchical_bootstrap
    import inspect
    repetitions=inspect.signature(hierarchical_bootstrap).parameters['n_boot'].default
    add('bootstrap_repetitions',repetitions,'src/nbv/evaluation.py','hierarchical_bootstrap.n_boot default')
    from nbv.evaluation import nest,nested_mean,paired_difference
    paired=nested_mean(paired_difference(nest(groups['sarsa/linear'],'normalized_cycle_time'),nest(groups['myopic/linear'],'normalized_cycle_time')))
    assert paired==verdict['H-forward']['mean_difference'],'paired difference and stored verdict differ'
    add('forward',paired,'results/registry.csv','paired canonical test tail variants',derived=True,formula='nested mean of per-variant SARSA seed mean minus myopic seed mean')
    add('forward_margin',.02+verdict['H-forward']['mean_difference'],'results/test_v1.json','H-forward.mean_difference',places=15,derived=True,formula='0.02 + mean_difference')
    add('shaping_auc_percent',100*verdict['H-shaping']['relative_auc_range'],'results/test_v1.json','H-shaping.relative_auc_range',places=6,derived=True,formula='100 * relative_auc_range')

    text=(root/'results/test_v1.md').read_text(encoding='utf-8')
    def table_rows(section):
        for line in section.splitlines():
            if line.startswith('| '):
                cells=[c.strip() for c in line.strip('|').split('|')]
                if cells[0] in LABELS:yield cells
    def cell(key, token, source_key):
        # Keep the original six-decimal endpoints; do not resample or infer intervals.
        rendered=token.replace(', ', '; ').replace('.', ',')
        claims.append(dict(key=key, raw=token, text=rendered,source='results/test_v1.md',source_key=source_key,derived=False,formula=None))
        return rendered
    part_rows=list(table_rows(text.split('Per part:',1)[1].split('Split summary:',1)[0]))
    values['table.parts']='\n\n'.join(
        f'#### Test parçası {part}\n\n| Yöntem | Normalize süre | Süre (s) | Başarı |\n|---|---:|---:|---:|\n'+'\n'.join(
            '| '+LABELS[label]+' | '+' | '.join(cell(f'part.{part}.{label}.{j}',token,f'primary part {part}, {label}') for j,token in enumerate(cells[2:]))+' |'
            for cells in part_rows for label in [cells[0]] if cells[1]==part)
        for part in sorted({c[1] for c in part_rows}, key=int))
    summary_rows=list(table_rows(text.split('Split summary:',1)[1].split('Hypotheses and pretest predictions:',1)[0]))
    lines=['| Yöntem | Normalize süre | Süre (s) | Başarı |','|---|---:|---:|---:|']
    for cells in summary_rows:
        label=cells[0]
        assert f'{_mean(groups[label],"normalized_cycle_time"):.6f}'==cells[2].split()[0], 'stored summary and registry differ'
        lines.append('| '+LABELS[label]+' | '+' | '.join(cell(f'summary.{label}.{j}',v,'split '+label) for j,v in enumerate(cells[2:]))+' |')
    values['table.summary']='\n'.join(lines)

    lines=['| Test parçası | H1′ eşiği | Kapanan boşluk payı | SARSA − MCTS | SARSA − miyop |',
           '|---|---:|---:|---:|---:|']
    for part,threshold in verdict['H1']['thresholds'].items():
        main=[r for r in groups['sarsa/linear'] if str(r['part'])==part]
        search=[r for r in groups['mcts'] if str(r['part'])==part]
        difference=_mean(main,'normalized_cycle_time')-_mean(search,'normalized_cycle_time')
        cells=[add('h1.threshold.'+part,threshold,'results/test_v1.json','H1.thresholds.'+part,places=6),
               add('h1.gap.'+part,verdict['H1']['gap_closed'][part],'results/test_v1.json','H1.gap_closed.'+part,places=6),
               add('h1.mcts.'+part,difference,'results/registry.csv','test part '+part,places=6,derived=True,formula='SARSA part mean - MCTS part mean'),
               add('forward.'+part,verdict['H-forward']['paired_difference'][part],'results/test_v1.json','H-forward.paired_difference.'+part,places=6)]
        lines.append('| '+part+' | '+' | '.join(cells)+' |')
    values['table.hypothesis_parts']='\n'.join(lines)
    lines=['| Potansiyel | Doğrulama öğrenme alanı | Test süresi − doğrusal |','|---|---:|---:|']
    for phi,area in verdict['H-shaping']['auc'].items():
        a=add('shaping.area.'+phi,area,'results/test_v1.json','H-shaping.auc.'+phi,places=6)
        delta=add('shaping.final.'+phi,verdict['H-shaping']['final_difference'][phi],'results/test_v1.json','H-shaping.final_difference.'+phi,places=6)
        lines.append('| '+{'linear':'Doğrusal','none':'Yok','convex':'Dışbükey','value_estimate':'Değer tahmini'}[phi]+' | '+a+' | '+delta+' |')
    values['table.shaping']='\n'.join(lines)
    for label in ('sarsa/linear','sarsa/value_estimate'):
        records=[]
        for row,extra in extras:
            if row['status']=='canonical' and extra.get('tag')=='p002train' and extra.get('weight_kind')=='tail' and row['method']+'/'+extra['phi']==label:
                records.extend(extra['episode_metrics'])
        add('gap.'+label,_mean(records,'normalized_cycle_time')-_mean(groups[label],'normalized_cycle_time'),
            'results/registry.csv','canonical train/test tail '+label,places=6,derived=True,formula='train nested mean - test nested mean')
    ceiling=(root/'results/raw_surface_ceiling_v1.md').read_text(encoding='utf-8')
    for name in ('train','val','test'):
        value=re.search(r'\| '+name+r' \| ([^|]+) \|',ceiling).group(1).strip()
        add('surface.'+name,value,'results/raw_surface_ceiling_v1.md',name)

    lines=['| Politika | Ortalama ms/karar | Medyan ms | Karar sayısı |','|---|---:|---:|---:|']
    for label, t in timing['timings'].items():
        cells=[add(f'timing.{label}',t['planning_time_ms_per_decision'],'results/registry.csv','planning timings.'+label,places=6),
               add(f'timing.median.{label}',t['median_ms'],'results/registry.csv','planning median.'+label,places=6),
               add(f'timing.calls.{label}',t['decisions'],'results/registry.csv','planning decisions.'+label)]
        lines.append('| '+LABELS[label]+' | '+' | '.join(cells)+' |')
    values['table.timing']='\n'.join(lines)
    add('timing_speedup',timing['timings']['mcts']['planning_time_ms_per_decision']/timing['timings']['sarsa/value_estimate']['planning_time_ms_per_decision'],
        'results/registry.csv','planning timings',places=1,derived=True,formula='MCTS mean ms / value-estimate SARSA mean ms')

    lines=['| Yöntem | Etkin adım | İz / düzenlileştirme | Keşif |','|---|---:|---|---|']
    for method in ('sarsa','regq','myopic'):
        setting=selected[method]['setting'];alpha=setting['alpha']/(11 if method=='sarsa' else 1)
        extra='λ = '+str(setting['lam']) if method=='sarsa' else 'η = '+str(setting['eta']) if method=='regq' else 'Önyükleme yok'
        a=add('alpha.'+method,alpha,'results/selected_settings_v1.json',method+'.setting.alpha',places=9,derived=method=='sarsa',formula='grid alpha / 11' if method=='sarsa' else None)
        lines.append('| '+{'sarsa':'SARSA(λ)','regq':'RegQ','myopic':'Miyop öğrenici'}[method]+' | '+a+' | '+extra.replace('.',',')+' | Sabit ε = 0,05 |')
    values['table.settings']='\n'.join(lines)

    attribution=_load(root,'provenance/part_sources.json')
    lines=['| Parça | Yazar | Bölme | Lisans | Sürüm / kaynak kontrolü | Değişiklikler |','|---|---|---|---|---|---|']
    for p in split['parts']:
        check=next(c for c in attribution if c['file_id']==p['file_id'])
        family='CC BY-SA' if p['share_alike'] else 'CC BY'
        split_name={'train':'Eğitim','val':'Doğrulama','test':'Test'}[p['split']]
        version=check['version'] or check['version_status']
        confirmation=check.get('source_confirmation')
        if confirmation:
            if confirmation['source_url'] != p['thing_url']:
                raise ValueError('license confirmation source mismatch')
            family=confirmation['license_label']
            version='Depo sahibi doğruladı ('+confirmation['confirmed_on']+')'
        source_cell=f'[{version}]({p["thing_url"]})'
        if confirmation:
            source_cell+=f' · [lisans]({confirmation["license_url"]})'
        changes='Ölçekleme, hizalama, örnekleme ve varyantlar'
        if p['split']=='val':changes+='; görünürlük türevi ve sadeleştirilmemiş gösterim mesh’i'
        else:changes+='; ham mesh dağıtılmıyor'
        lines.append(f'| {p["name"]} | {p["author"]} | {split_name} | {family} | {source_cell} | {changes} |')
    values['table.attribution']='\n'.join(lines)

    collision=_load(root,'results/collision_audit_v1.json')
    lines=['| Yöntem | VAL çarpışan geçiş | Train çarpışan geçiş |','|---|---:|---:|']
    for label in ('greedy_cost','mcts','ilp_tsp','joint_opt','sarsa/linear','sarsa/value_estimate','regq/linear','myopic/linear'):
        cells=[]
        for split_name in ('val','train'):
            records=[r for r in collision['by_part_method'] if r['split']==split_name and r['method']==label]
            num=sum(r['colliding_transfers'] for r in records);den=sum(r['transfers'] for r in records)
            cells.append(add(f'collision.{split_name}.{label}',100*num/den,'results/collision_audit_v1.json','by_part_method',places=2,derived=True,formula='100 * sum(colliding_transfers) / sum(transfers)')+'%')
        lines.append('| '+LABELS[label]+' | '+' | '.join(cells)+' |')
    values['table.collision']='\n'.join(lines)
    add('unmeasured_train',collision['optional_unmeasured_episodes'],'results/collision_audit_v1.json','optional_unmeasured_episodes')

    finding=(root/'results/phase1_findings_v1.md').read_text(encoding='utf-8')
    values['table.cross_split']='\n'.join(
        line.replace('| SARSA Phi |','| SARSA potansiyeli |').replace('Train','Eğitim').replace('VAL','Doğrulama').replace('Test','Test')
        .replace('| linear |','| Doğrusal |').replace('| none |','| Yok |').replace('| convex |','| Dışbükey |').replace('| value_estimate |','| Değer tahmini |').replace('| MCTS reference |','| MCTS |')
        .replace('.',',') for line in finding.splitlines() if line.startswith('|'))
    claims.append(dict(key='table.cross_split',raw=finding,text=values['table.cross_split'],source='results/phase1_findings_v1.md',source_key='stored cross-split table',derived=False,formula=None))
    references=_load(root,'provenance/references.json')
    citation_order=[]
    def cite(match):
        key=match.group(1)
        if key not in citation_order:citation_order.append(key)
        if key not in references:raise ValueError('unverified or missing reference: '+key)
        return '['+str(citation_order.index(key)+1)+']'
    template=re.sub(r'\[\[cite:([^\]]+)\]\]',cite,template)
    values['references']='\n\n'.join(f'[{i}] {references[key]["authors"]} ({references[key]["year"]}). [{references[key]["title"]}]({references[key]["url"]}). {references[key]["venue"]}.' for i,key in enumerate(citation_order,1))
    def replace(match):
        key=match.group(1)
        if key not in values:raise ValueError('missing source claim: '+key)
        return values[key]
    result=re.sub(r'\{\{([^}]+)\}\}',replace,template).rstrip()+'\n'
    return result,claims


def verify(text=None, root=ROOT):
    root=Path(root)
    expected,claims=render(root)
    actual=text if text is not None else (root/'README.md').read_text(encoding='utf-8')
    if actual!=expected:raise ValueError('README does not match source-derived manuscript')
    visible=re.sub(r'```.*?```','',actual,flags=re.S)
    visible=re.sub(r'!\[[^\]]*\]\([^)]*\)','',visible)
    visible=visible.replace('(docs/PROTOCOL.md)','')
    if re.search(r'\bD-\d{3}\b|\b(?:plan|slice|exec[- ]plan)\b|\b(?:ENV|PROTOCOL)\s+v\d|\b[0-9a-f]{40}\b|\b(?:agent|ajan)\b',visible,re.I):
        raise ValueError('internal identifier in public manuscript')
    metadata=_load(root,'provenance/publication.json')['weights']
    payload=(root/'weights/linear_learners.json').read_bytes()
    if hashlib.sha256(payload).hexdigest()!=metadata['sha256'] or len(payload)!=metadata['bytes']:
        raise ValueError('published weights do not match the recorded hash/size')
    return dict(claims=len(claims),source_mismatches=0,weights_sha256=metadata['sha256'],bootstrap_resampling=False,policy_evaluation=False)
