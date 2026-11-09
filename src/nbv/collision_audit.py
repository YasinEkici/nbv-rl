"""Read-only D-042 trajectory diagnosis. No planner, policy, training or test input access."""
from __future__ import annotations
import json
import hashlib
import csv
from collections import defaultdict
from pathlib import Path
import time
import numpy as np
from nbv import collision,parts,robot


class WaypointCollision(RuntimeError):
    def __init__(self,event):
        self.event=event
        super().__init__('Recorded IK waypoint collision: '+json.dumps(event,allow_nan=False))


def allowed_part(fid,manifest=None):
    manifest=manifest or parts.load_yaml('configs/split_v1.yaml')
    entries=[p for p in manifest['parts'] if int(p['file_id'])==int(fid)]
    if len(entries)!=1 or entries[0]['split'] not in ('train','val'):
        raise ValueError('closed test/unknown part: collision audit permits train/VAL only')
    return entries[0]['split']


def sample_joint_segment(start,end,*,max_step_deg=1.):
    a,b=np.asarray(start,dtype=np.float64),np.asarray(end,dtype=np.float64)
    if a.shape!=(6,) or b.shape!=(6,) or not np.isfinite([a,b]).all() or max_step_deg<=0:
        raise ValueError('finite executed six-joint endpoints and positive step required')
    count=max(1,int(np.ceil(np.abs(b-a).max()/np.deg2rad(max_step_deg))))
    fractions=np.linspace(0.,1.,count+1)
    points=a[None,:]+fractions[:,None]*(b-a)[None,:]
    points[0],points[-1]=a,b
    return points,fractions


def _values(clearance):
    result=[]
    for body,pair in clearance.items():
        for obstacle,value in zip(('part','table'),pair):
            value=float(value)
            if np.isnan(value) or value==-np.inf:raise ValueError('invalid geometry clearance')
            result.append(dict(body=body,obstacle=obstacle,clearance_mm=value))
    return result


def require_clear_waypoint(clearance,context):
    hits=[v for v in _values(clearance) if v['clearance_mm']<0.]
    if hits:
        worst=min(hits,key=lambda v:v['clearance_mm'])
        raise WaypointCollision(context|worst|{'penetration_mm':-worst['clearance_mm']})


def inspect_segment(start,end,pose_clearance,*,max_step_deg=1.,near_mm=10.):
    q,f=sample_joint_segment(start,end,max_step_deg=max_step_deg)
    samples=[_values(pose_clearance(p)) for p in q]
    keys=[(v['body'],v['obstacle']) for v in samples[0]]
    values=np.asarray([[v['clearance_mm'] for v in s] for s in samples])
    intervals=[]
    for col,(body,obstacle) in enumerate(keys):
        hits=np.flatnonzero(values[:,col]<0.)
        for group in np.split(hits,np.flatnonzero(np.diff(hits)>1)+1):
            if not len(group):continue
            worst=int(group[np.argmin(values[group,col])])
            intervals.append(dict(body=body,obstacle=obstacle,start_fraction=float(f[group[0]]),end_fraction=float(f[group[-1]]),
                first_sample=int(group[0]),last_sample=int(group[-1]),sample_count=len(group),
                worst_fraction=float(f[worst]),worst_q_rad=q[worst].tolist(),
                clearance_mm=float(values[worst,col]),penetration_mm=float(-values[worst,col])))
    worst=min(intervals,key=lambda v:v['clearance_mm']) if intervals else None
    minimum=float(values.min())
    return dict(collided=bool(intervals),near=minimum<near_mm,minimum_clearance_mm=minimum,
        endpoint_collision=bool((values[[0,-1]]<0).any()),worst=worst,intervals=intervals,pose_samples=len(q))


class SingleThreadPartDistance(collision.PartDistance):
    """Identical Open3D query with one worker per part (parallelism is across parts)."""
    def signed(self,points):
        query=collision.o3d.core.Tensor(points.astype(np.float32))
        return self.scene.compute_signed_distance(query,nthreads=1).numpy().astype(float)


class NominalCollisionModel:
    def __init__(self,fid):
        self.split=allowed_part(fid);self.part=int(fid)
        cfg=parts.load_yaml('configs/parts_v2.yaml');self.robot_cfg=parts.load_yaml('configs/robot_v2.yaml')
        self.collision_cfg=parts.load_yaml('configs/collision_v1.yaml')
        g=cfg['geometry'];self.mesh=parts.load_base(Path(cfg['paths']['raw'])/f'{fid}.ply',g['longest_mm'],g['max_edge_mm'])
        self.model=robot.make_robot(self.robot_cfg);self.geom=collision.ArmGeometry.from_config(self.collision_cfg)
        self.distance=SingleThreadPartDistance(self.mesh);self.box=self.geom.box_points();self.cache={};self.calls=0

    def __call__(self,q):
        q=np.asarray(q,dtype=np.float64)
        if q.shape!=(6,) or not np.isfinite(q).all():raise ValueError('finite joint state required')
        key=q.tobytes()
        if key not in self.cache:
            self.cache[key]=collision.pose_clearance(self.model,q,self.distance,self.geom,self.box);self.calls+=1
        return self.cache[key]


def check_waypoints():
    """Gate on all allowed saved reachable IK samples before any dense sweep."""
    manifest=parts.load_yaml('configs/split_v1.yaml');records=[];started=time.perf_counter()
    for p in manifest['parts']:
        if p['split'] not in ('train','val'):continue
        fid=p['file_id'];model=NominalCollisionModel(fid)
        path=Path(f'data/cache/robot/nominal/{fid}_n48.npz')
        with np.load(path) as z:qs=z['q'];reachable=z['reachable'].astype(bool)
        minimum=np.inf;worst=None
        try:
            for action in np.flatnonzero(reachable):
                for index,q in enumerate(qs[action]):
                    context=dict(part=fid,split=p['split'],pass_index=int(action),sample=index,q_rad=q.tolist(),phase='saved_ik_sample')
                    clear=model(q);require_clear_waypoint(clear,context)
                    v=min(_values(clear),key=lambda v:v['clearance_mm'])
                    if v['clearance_mm']<minimum:minimum=v['clearance_mm'];worst=context|v
        except WaypointCollision as exc:
            report=dict(complete=False,stopped=True,reason='negative clearance at original saved IK sample',event=exc.event,
                completed_parts=records,no_policy_calls=True,test_access=False,geometry='D-042 nominal CAD',wall_s=time.perf_counter()-started)
            Path('results/collision_waypoint_gate_v1.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
            raise
        r=dict(part=fid,split=p['split'],checked=int(reachable.sum()*qs.shape[1]),minimum_clearance_mm=float(minimum),worst=worst,
               geometry_calls=model.calls,mesh_triangles=len(model.mesh.faces))
        records.append(r);print(f'Waypoint gate {fid}: {r["checked"]} poses, min {minimum:.6f} mm',flush=True)
    report=dict(complete=True,stopped=False,parts=records,checked=sum(r['checked'] for r in records),no_policy_calls=True,
        test_access=False,geometry='D-042 nominal CAD, original subdivision and exact collision.pose_clearance',wall_s=time.perf_counter()-started)
    Path('results/collision_waypoint_gate_v1.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    return report


def episode_segments(record,home):
    previous=0.
    for index,(step,trace) in enumerate(zip(record['steps'],record['joint_trace'])):
        q=np.asarray(trace['q_samples'],dtype=float)
        finish=float(trace['elapsed_before_home']);scan_start=finish-step['t_scan']
        base=dict(step=index,pass_index=step['pass'])
        yield base|dict(phase='transfer',start=trace['q_start'],end=q[0].tolist(),time_s=[previous,scan_start])
        for j in range(4):
            yield base|dict(phase='intra_pass',subsegment=j,start=q[j].tolist(),end=q[j+1].tolist(),
                            time_s=[scan_start+step['t_scan']*j/4,scan_start+step['t_scan']*(j+1)/4])
        previous=finish
    if record['steps']:
        yield dict(step=len(record['steps']),pass_index=None,phase='home',start=record['joint_trace'][-1]['q_end'],
                   end=np.asarray(home).tolist(),time_s=[previous,previous+record['steps'][-1]['t_home']])


def summarize_events(episodes):
    groups=defaultdict(list)
    for e in episodes:groups[(e['split'],e['weight_kind'],e['part'],e['method'])].append(e)
    rows=[]
    for (split,kind,part,method),records in sorted(groups.items()):
        homes=sum(e.get('phase_counts',{}).get('home',[0,0])[0] for e in records)
        home_hits=sum(e.get('phase_counts',{}).get('home',[0,0])[1] for e in records)
        transfers=sum(e['transfer_count'] for e in records)+homes;hits=sum(e['colliding_transfers'] for e in records)+home_hits
        depths=[d for e in records for d in e['penetrations_mm']]
        rows.append(dict(split=split,weight_kind=kind,part=part,method=method,episodes=len(records),
            transfers=transfers,colliding_transfers=hits,colliding_transfer_rate=hits/transfers if transfers else 0.,
            start_interpass_transfers=transfers-homes,colliding_start_interpass_transfers=hits-home_hits,
            home_transfers=homes,colliding_home_transfers=home_hits,
            colliding_episodes=sum(e['any_collision'] for e in records),
            colliding_episode_rate=sum(e['any_collision'] for e in records)/len(records),
            colliding_motion_segments=len(depths),penetration_mm=dict(zip(('min','median','p95','max'),
                np.quantile(depths,[0,.5,.95,1]).tolist())) if depths else None))
    return rows


def prepare_records(destination='scratch/p004b/audit_records.jsonl'):
    """Extend verified VAL traces with train sequences through the same D-069 replay."""
    from nbv.visualization.replay import validated_records,source_records,replay_episode,digest
    from nbv.env import NBVEnv,ENV_VERSION
    records,validation=validated_records();sources=dict(validation['sources'])
    records=[e|dict(weight_kind='baseline' if e['verification']['stage']=='A' else 'tail') for e in records]
    manifest=parts.load_yaml('configs/split_v1.yaml')
    variants=[v['variant_id'] for v in manifest['variants'] if v['split']=='train']
    rows=list(csv.DictReader(open('results/registry.csv',encoding='utf-8')))
    rows=[r for r in rows if r['status']=='canonical' and r['part_split']=='train' and r['env_version']==ENV_VERSION
          and (r['plan']=='001' or json.loads(r['extra_json']).get('tag')=='p002train')]
    if len(rows)!=145:raise ValueError('expected 25 canonical train baseline + 120 learner evaluation sources')
    reference={e['variant']:e['cycle_time'] for e in source_records(next(r for r in rows if r['method']=='ilp_tsp'))}
    if set(reference)!=set(variants):raise ValueError('train reference variant set')
    env=NBVEnv(variants)
    for i,row in enumerate(rows):
        extra=json.loads(row['extra_json']);baseline=row['plan']=='001'
        stored_records=source_records(row)
        if len(stored_records)!=40 or {r['variant'] for r in stored_records}!=set(variants):raise ValueError('train source variant set')
        sources[row['run_id']]=dict(sha256=digest(Path('outputs')/row['run_id']/'episodes.jsonl'),commit=row['commit'],
            method=row['method'],seed=row['seed'])
        for stored in stored_records:
            allowed_part(stored['part'],manifest)
            fresh=replay_episode(env,stored,reference[stored['variant']],stage='A' if baseline else 'B')
            fresh.update(source_run=row['run_id'],method=row['method'] if baseline else f"{row['method']}/{extra['phi']}",
                seed=int(row['seed'] or 0),split='train',weight_kind='baseline' if baseline else extra['weight_kind'])
            records.append(fresh)
        if (i+1)%10==0:print(f'Train recorded-sequence oracle {i+1}/{len(rows)} sources',flush=True)
    target=Path(destination);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(''.join(json.dumps(e,allow_nan=False)+'\n' for e in records),encoding='utf-8')
    metadata=dict(records=len(records),val=1095,train=len(records)-1095,no_policy_calls=True,no_test_access=True,
                  sources=sources,replay_sha256=digest(target),registry_sha256=digest('results/registry.csv'))
    target.with_suffix('.sources.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return metadata


def segment_id(part,start,end):
    return hashlib.sha256(str(part).encode()+np.asarray([start,end],dtype='<f8').tobytes()).hexdigest()


def query_equivalence_part(fid):
    model=NominalCollisionModel(fid)
    original=collision.PartDistance.__new__(collision.PartDistance)
    original.scene=model.distance.scene;original.vertices=model.distance.vertices
    with np.load(f'data/cache/robot/nominal/{fid}_n48.npz') as z:
        saved=z['q'][z['reachable'].astype(bool)]
    home=np.deg2rad(model.robot_cfg['q_home_deg']);qs=list(saved.reshape(-1,6))
    for end in saved[:6,0]:
        dense,_=sample_joint_segment(home,end)
        qs.extend(dense[np.unique(np.linspace(0,len(dense)-1,12).astype(int))])
    start=time.perf_counter()
    for index,q in enumerate(qs):
        a=collision.pose_clearance(model.model,q,original,model.geom,model.box)
        b=collision.pose_clearance(model.model,q,model.distance,model.geom,model.box)
        if a.keys()!=b.keys() or np.asarray(list(a.values()),dtype='<f8').tobytes()!=np.asarray(list(b.values()),dtype='<f8').tobytes():
            raise ValueError(f'Raycast threading changed clearance bits: part {fid}, sample {index}')
    return dict(part=fid,poses=len(qs),all_body_obstacle_clearances_bit_identical=True,wall_s=time.perf_counter()-start)


def verify_query_equivalence(workers=11):
    from concurrent.futures import ProcessPoolExecutor
    manifest=parts.load_yaml('configs/split_v1.yaml')
    ids=[p['file_id'] for p in manifest['parts'] if p['split'] in ('train','val')]
    with ProcessPoolExecutor(max_workers=workers) as pool:results=list(pool.map(query_equivalence_part,ids))
    result=dict(complete=True,comparison='Open3D default query parallelism vs nthreads=1; unchanged collision.pose_clearance',
                scope='Every saved reachable IK sample plus dense home-to-pass path samples; no test inputs',
                parts=results,poses=sum(r['poses'] for r in results),all_bit_identical=True)
    Path('results/collision_query_equivalence_v1.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def geometry_input_hashes(fid):
    from nbv.visualization.replay import digest
    allowed_part(fid);cfg=parts.load_yaml('configs/parts_v2.yaml')
    paths=['configs/parts_v2.yaml','configs/robot_v2.yaml','configs/collision_v1.yaml',
           f'data/cache/robot/nominal/{fid}_n48.npz',str(Path(cfg['paths']['raw'])/f'{fid}.ply')]
    return {p:digest(p) for p in paths}


def restore_geometries(path,fid,expected_sha,expected_inputs):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_sha:raise ValueError('geometry snapshot hash mismatch')
    saved=json.loads(raw)
    if saved['part']!=fid or saved['max_step_deg']!=1. or saved['near_mm']!=10.:
        raise ValueError('geometry snapshot identity/sampling mismatch')
    if saved['geometry_input_sha256']!=expected_inputs:raise ValueError('geometry snapshot input hashes mismatch')
    return saved['geometries']


def in_scope(record,scope):
    if record['split'] not in ('train','val'):return False
    if scope=='all':return True
    if scope=='val':return record['split']=='val'
    if scope=='train-primary':return record['split']=='train' and record['weight_kind'] in ('tail','baseline')
    if scope=='train-final':return record['split']=='train' and record['weight_kind']=='final'
    raise ValueError('unknown audit scope')


def audit_part(task):
    fid,records,directory,warm=task;model=NominalCollisionModel(fid);home=np.deg2rad(model.robot_cfg['q_home_deg'])
    geometries=restore_geometries(warm['path'],fid,warm['sha256'],geometry_input_hashes(fid)) if warm else {}
    reused=len(geometries);episodes=[];events=[];started=time.perf_counter()
    for number,record in enumerate(records):
        identity={k:record[k] for k in ('part','variant','method','seed','split','weight_kind','source_run')}
        identity['id']=f"{record['source_run']}:{record['variant']}"
        # Executed lifted samples retain the ENV v4 wrap/limit choices; never replace with nearest-angle poses.
        for step,trace in enumerate(record['joint_trace']):
            for index,q in enumerate(trace['q_samples']):
                require_clear_waypoint(model(q),identity|dict(step=step,sample=index,q_rad=q,phase='executed_ik_sample'))
        count=hits=0;depths=[];phase_counts=defaultdict(lambda:[0,0])
        for segment in episode_segments(record,home):
            key=segment_id(fid,segment['start'],segment['end'])
            if key not in geometries:
                geometries[key]=inspect_segment(segment['start'],segment['end'],model,near_mm=float(model.collision_cfg['near_mm']))
            result=geometries[key];phase=segment['phase'];phase_counts[phase][0]+=1
            if phase=='transfer':count+=1
            if result['collided']:
                if phase=='transfer':hits+=1
                phase_counts[phase][1]+=1;depths.append(result['worst']['penetration_mm'])
                events.append(identity|segment|dict(geometry_id=key,penetration_mm=result['worst']['penetration_mm'],
                    body=result['worst']['body'],obstacle=result['worst']['obstacle']))
        episodes.append(identity|dict(transfer_count=count,colliding_transfers=hits,any_collision=bool(depths),
                                     penetrations_mm=depths,phase_counts=dict(phase_counts)))
        if number%10==0:
            print(f'Audit {fid}: {number+1}/{len(records)} episodes, {len(geometries)} distinct paths, {model.calls} poses',flush=True)
        # Pose cache is only a speed aid; cap RAM without changing exact path-result cache.
        if len(model.cache)>20000:model.cache.clear()
    output=dict(part=fid,episodes=episodes,events=events,
        geometries={k:v for k,v in geometries.items() if v['collided']},
        unique_paths=len(geometries),geometry_calls=model.calls,reused_paths=reused,wall_s=time.perf_counter()-started)
    Path(directory,f'part_{fid}.json').write_text(json.dumps(output,allow_nan=False),encoding='utf-8')
    return dict(part=fid,episodes=len(episodes),events=len(events),wall_s=output['wall_s'])


def run_audit(source='scratch/p004b/audit_records.jsonl',workers=8,*,scope='all',warm_cache=None):
    from concurrent.futures import ProcessPoolExecutor,as_completed
    from nbv.visualization.replay import digest
    from nbv import evaluation
    measurement_state=evaluation.git_state()|dict(code_sha256=evaluation.code_hash())
    path=Path(source);metadata=json.loads(path.with_suffix('.sources.json').read_text(encoding='utf-8'))
    if digest(path)!=metadata['replay_sha256']:raise ValueError('audit recorded trace hash mismatch')
    gate=json.loads(Path('results/collision_waypoint_gate_v1.json').read_text(encoding='utf-8'))
    if not gate['complete'] or gate['stopped']:raise ValueError('saved waypoint gate required')
    equivalence=json.loads(Path('results/collision_query_equivalence_v1.json').read_text(encoding='utf-8'))
    if not equivalence['complete'] or not equivalence['all_bit_identical']:raise ValueError('query parallelism equality gate required')
    groups=defaultdict(list);split_manifest=parts.load_yaml('configs/split_v1.yaml')
    for line in path.read_text(encoding='utf-8').splitlines():
        e=json.loads(line);allowed_part(e['part'],split_manifest)
        if in_scope(e,scope):groups[int(e['part'])].append(e)
    expected={'all':6895,'val':1095,'train-primary':3400,'train-final':2400}[scope]
    if sum(map(len,groups.values()))!=expected:raise ValueError('audit scope episode count mismatch')
    metadata=metadata|dict(input_corpus_records=metadata['records'],records=expected,scope=scope)
    selected_sources={e['source_run'] for records in groups.values() for e in records}
    metadata['sources']={k:v for k,v in metadata['sources'].items() if k in selected_sources}
    print(f"Allowed trace inventory: {sum(map(len,groups.values()))} episodes / {len(groups)} parts",flush=True)
    warm={}
    if warm_cache:
        warm=json.loads(Path(warm_cache).read_text(encoding='utf-8'))['parts']
        for fid,item in warm.items():
            allowed_part(int(fid),split_manifest)
            if not Path(item['path']).resolve().is_relative_to(Path('scratch').resolve()):
                raise ValueError('warm geometry snapshots must be private scratch files')
    directory=Path('scratch/p004b/geometry')/scope;directory.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(audit_part,(fid,records,str(directory),warm.get(str(fid)))) for fid,records in groups.items()]
        try:
            for future in as_completed(futures):print('Part complete:',future.result(),flush=True)
        except BaseException:
            pool.terminate_workers()
            raise
    results=[json.loads((directory/f'part_{fid}.json').read_text(encoding='utf-8')) for fid in groups]
    episodes=[e for r in results for e in r['episodes']];events=[e for r in results for e in r['events']]
    geometries={k:v for r in results for k,v in r['geometries'].items()}
    report=dict(schema_version=1,complete=True,scope=scope,sampling_max_joint_degrees=1.,near_mm=10.,measurement_provenance=measurement_state,
        geometry='Unchanged D-042 nominal CAD/capsules/sensor box/table and exemptions; no self-collision/fixtures.',
        time_mapping='Path fractions mapped linearly to recorded segment times for display only; not certified actuator timing.',
        interval_definition='Contiguous colliding sampled poses; boundaries do not certify exact continuous contact onset.',
        penetration_distribution='Maximum penetration per colliding motion segment, retaining repeated episodes.',
        sources=metadata,waypoint_gate_sha256=digest('results/collision_waypoint_gate_v1.json'),
        query_parallelism='One Open3D query thread per part; processes across parts. Original pose_clearance unchanged.',
        query_equivalence_sha256=digest('results/collision_query_equivalence_v1.json'),
        episodes=episodes,events=events,geometries=geometries,by_part_method=summarize_events(episodes),
        geometry_work=[{k:r[k] for k in ('part','unique_paths','geometry_calls','reused_paths','wall_s')} for r in results])
    target=report_path(scope)
    target.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    finalize_audit(target)
    return report


def deepest_events(report,count=3):
    """Visualize distinct part/body/obstacle/pose states, including non-worst bodies."""
    seen_paths=set();states={}
    for event in report['events']:
        key=event['geometry_id']
        if key in seen_paths:continue
        seen_paths.add(key)
        for i,interval in enumerate(report['geometries'][key]['intervals']):
            state=(event['part'],interval['body'],interval['obstacle'],np.asarray(interval['worst_q_rad'],dtype='<f8').tobytes())
            candidate=event|dict(body=interval['body'],obstacle=interval['obstacle'],penetration_mm=interval['penetration_mm'],
                                 collision_interval=interval,interval_index=i)
            if state not in states or candidate['penetration_mm']>states[state]['penetration_mm']:states[state]=candidate
    return sorted(states.values(),key=lambda e:(-e['penetration_mm'],e['geometry_id'],e['interval_index']))[:count]


def report_path(scope):
    suffix={'all':'','val':'_val','train-primary':'_train','train-final':'_train_final'}[scope]
    return Path(f'results/collision_audit{suffix}_v1.json')


def write_report(report):
    """All counts and distribution cells are generated from the audit's event ledger."""
    lines=['# Recorded-path collision audit (D-071)',
        '', 'Descriptive train/VAL geometry measurement only. No policy calls, closed-test input access, trajectory repair, '
        'quality-metric or hypothesis changes. Original D-042 nominal geometry and exemptions are retained.', '',
        f"Recorded episodes: {len(report['episodes'])}; saved IK waypoint gate: passed. Maximum joint sample spacing: "
        f"{report['sampling_max_joint_degrees']:g} degree; near-contact threshold: {report['near_mm']:g} mm.", '',
        'Transfers include start/inter-pass movements and return home. Four in-pass subsegments are separately checked '
        'and enter the any-collision episode flag; they are excluded from the transfer denominator. Penetration quantiles use '
        'the deepest sampled penetration per colliding motion segment, with repeated episode occurrences retained. '
        'Primary tail policies/baselines and final-weight train policies remain separate. A safe discrete sample is not '
        'a continuous clearance guarantee. Display times linearly map path fractions to recorded time; they do not '
        'certify the physical acceleration law.', '',
        '| Split | Weights | Part | Method | Episodes | Transfers hit / all | Transfer rate | Episodes with collision | Penetration min / median / p95 / max (mm) |',
        '|---|---|---|---|---:|---:|---:|---:|---|']
    overall=summarize_events([e|{'part':'all'} for e in report['episodes']])
    for row in overall+report['by_part_method']:
        depth=row['penetration_mm'];text='none' if depth is None else ' / '.join(f'{v:.3f}' for v in depth.values())
        lines.append(f"| {row['split']} | {row['weight_kind']} | {row['part']} | {row['method']} | {row['episodes']} | "
            f"{row['colliding_transfers']} / {row['transfers']} | {row['colliding_transfer_rate']:.2%} | "
            f"{row['colliding_episodes']} / {row['episodes']} ({row['colliding_episode_rate']:.2%}) | {text} |")
    lines+=['', '## Deepest distinct geometric events', '',
        '| Part | Variant | Method | Phase | Step | Body | Obstacle | Penetration (mm) | Source |',
        '|---|---|---|---|---:|---|---|---:|---|']
    for event in deepest_events(report):
        lines.append(f"| {event['part']} | {event['variant']} | {event['method']} | {event['phase']} | {event['step']} | "
            f"{event['body']} | {event['obstacle']} | {event['penetration_mm']:.6f} | {event['source_run']} |")
    lines+=['', '## Interpretation and scope', '',
        'Short joint-space paths are not collision-free paths. Per-method rates describe a potential source of '
        'comparison bias in the frozen estimated-cycle-time model; they do not establish a new quality ranking or '
        'change any reported hypothesis. Compare rates within each part and split rather than pooling different '
        'part geometries or final and tail weights. The ledger identifies every affected segment/body/obstacle.', '',
        'D-021/D-042 checked only saved scan poses, not transfers or the intervals between scan samples. A future '
        'ENV v5 proposal is a retract waypoint plus densely checked motion and recomputed time matrices. That '
        'requires regenerated baseline and learning results; none of it is applied here.', '',
        '[Machine-readable event ledger](collision_audit_v1.json) · [Original waypoint gate](collision_waypoint_gate_v1.json)', '']
    target=report_path(report.get('scope','all')).with_suffix('.md')
    lines.append(f"Scope: {report.get('scope','all')}. The source corpus may include other scopes; only the listed episodes enter these rates.")
    target.write_text('\n'.join(lines),encoding='utf-8')


def finalize_audit(path='results/collision_audit_v1.json',*,measurement_commit=None):
    """Validate the measured ledger and regenerate aggregates, never geometry or episodes."""
    from collections import Counter
    from nbv.visualization.replay import digest
    from nbv import evaluation
    report=json.loads(Path(path).read_text(encoding='utf-8'))
    if not report['complete'] or len(report['episodes'])!=report['sources']['records']:
        raise ValueError('incomplete collision measurement')
    counts=Counter((e['id'],e['phase']) for e in report['events'])
    for e in report['episodes']:
        for phase,(total,hits) in e['phase_counts'].items():
            if not 0<=hits<=total or counts[(e['id'],phase)]!=hits:raise ValueError('event/phase count mismatch')
        if bool(sum(v[1] for v in e['phase_counts'].values()))!=e['any_collision']:raise ValueError('episode collision flag mismatch')
    report['by_part_method']=summarize_events(report['episodes'])
    report['transfer_definition']='Start/inter-pass plus return home; raw episode transfer_count field counts start/inter-pass only, phase_counts.home is added by the aggregator.'
    report['event_schema']='events identifies every colliding motion-segment occurrence with episode/step/phase/times/endpoints; join geometry_id to geometries.intervals for EVERY contiguous body/obstacle collision, not just the worst body copied onto the occurrence.'
    report['collision_interval_occurrences']=sum(len(report['geometries'][e['geometry_id']]['intervals']) for e in report['events'])
    report['deepest_visualization_definition']='Three distinct (part, body, obstacle, exact joint pose) states, ranked by penetration over all body/obstacle intervals; repeated episode occurrences at an identical state are represented once for screenshots only.'
    inputs=['configs/parts_v2.yaml','configs/robot_v2.yaml','configs/collision_v1.yaml','configs/split_v1.yaml']
    cfg=parts.load_yaml('configs/parts_v2.yaml')
    for p in sorted({e['part'] for e in report['episodes']}):
        allowed_part(p);inputs.extend([str(Path(cfg['paths']['raw'])/f'{p}.ply'),f'data/cache/robot/nominal/{p}_n48.npz'])
    report['geometry_input_sha256']={p:digest(p) for p in inputs}
    report['measurement_core_commit']=report.get('measurement_core_commit') or report.get('measurement_provenance',{}).get('commit') or measurement_commit
    if not report['measurement_core_commit']:raise ValueError('original measurement core commit must be specified')
    report['aggregation_provenance']=evaluation.git_state()|dict(code_sha256=evaluation.code_hash())
    Path(path).write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8');write_report(report)
    return report
