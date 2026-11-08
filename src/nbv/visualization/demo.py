"""Lossless VAL-only viewer bundle, exact replay gates and decimal 12 MB limit."""
import base64
import json
import math
from pathlib import Path
import tempfile

import numpy as np
from jsonschema import Draft202012Validator

from nbv.evaluation import git_state
from nbv.visualization import assets
from nbv.visualization.replay import digest, require_exact, validated_records, SUMMARY_FIELDS, STEP_FIELDS
from nbv.visualization.styles import STYLES

VAL_PARTS=(271875,591247,37275)
METHODS=('greedy_cost','mcts','ilp_tsp','joint_opt','sarsa/linear','sarsa/value_estimate','regq/linear')
COORDINATES={
    'frame':'Part/world frame: nominal bbox bottom-center at origin, x/y horizontal, z upward.',
    'mesh':'normalized nominal display mesh',
    'points':'variant v0 positions and face normals in original 20,000-sample correspondence order',
    'warning':'Mesh nominal, points variant v0; deviation up to approximately 1.5 mm.',
    'warning_tr':'mesh nominal, noktalar varyant v0; sapma \u2264 ~1,5 mm',
    'length_unit':'mm','angle_unit':'rad','time_unit':'s',
}


def encode_array(value,units):
    a=np.asarray(value)
    if a.dtype.kind=='f': dtype='<f8'
    elif a.dtype.kind in 'ub' and a.dtype.itemsize==1: dtype='|u1'
    elif a.dtype.kind in 'ui' and a.size and a.min()>=0 and a.max()<=np.iinfo(np.uint32).max: dtype='<u4'
    else: raise ValueError('unsupported array dtype; no silent lossy encoding')
    if a.dtype.kind=='f' and (a.dtype.itemsize>8 or not np.isfinite(a).all()): raise ValueError('finite binary64 array required')
    a=np.ascontiguousarray(a,dtype=dtype)
    return dict(dtype=dtype,shape=list(a.shape),byte_order='not-applicable' if dtype=='|u1' else 'little',
                order='C',encoding='base64',data=base64.b64encode(a.tobytes()).decode('ascii'),units=units)


def decode_array(item):
    dtype=item['dtype']
    if dtype not in ('<f8','<u4','|u1') or item['order']!='C' or item['encoding']!='base64': raise ValueError('array encoding')
    if item['byte_order']!=('not-applicable' if dtype=='|u1' else 'little'): raise ValueError('byte order')
    shape=item['shape']
    if any(type(n) is not int or n<0 for n in shape): raise ValueError('array shape')
    raw=base64.b64decode(item['data'],validate=True)
    if len(raw)!=math.prod(shape)*np.dtype(dtype).itemsize: raise ValueError('array byte length')
    result=np.frombuffer(raw,dtype=dtype).reshape(shape)
    if not np.isfinite(result).all(): raise ValueError('finite array required')
    return result


def pack_visibility(bits):
    bits=np.asarray(bits,dtype=bool)
    return encode_array(np.packbits(bits,axis=1,bitorder='big'),'measured:boolean')|dict(
        logical_shape=list(bits.shape),bit_order='big',pack_axis=1,bit_count=int(bits.size),
        row_padding_bits=int((-bits.shape[1])%8),definition='Pass-major rows; first sample = most significant bit of first byte. Trailing row padding is zero.')


def attribution(source):
    version=source.get('license_version')
    return dict(dataset='Thingi10K',file_id=source['file_id'],name=source['name'],author=source['author'],
        source_url=source['thing_url'],license=source['license'],license_version=version,
        version_status='recorded' if version else 'version not recorded',
        version_status_tr='kayitli' if version else 's\u00fcr\u00fcm kay\u0131tl\u0131 de\u011fil',
        official_specific_license_url=source.get('license_url') if version else None,
        official_license_overview='https://creativecommons.org/licenses/',share_alike=bool(source['share_alike']),
        viewer_requirement='Show title, author, source link, recorded license family/version status and changes; BY-SA derivatives retain the same license. Do not invent a version-specific deed.',
        changes={'nominal_mesh':['Uniformly scaled to 200 mm longest side; bbox bottom-center aligned to origin','No display simplification; original nominal triangles retained'],
            'variant_points':['20,000 area-weighted surface samples with saved face/barycentric correspondence','Stored variant-v0 deviations; positions and face normals from variant geometry'],
            'visibility':['Derived pass-by-sample visibility measured bits on variant v0; no recomputation']})


def episode_document(record,q_home,manifest_sha):
    verify=record['verification']
    if record['split']!='val' or not verify['exact'] or verify['stage'] not in ('A','B'): raise ValueError('unverified VAL episode')
    if not set(SUMMARY_FIELDS)<=set(verify['matched_fields']): raise ValueError('incomplete partial oracle')
    if verify['stage']=='A' and set(verify['matched_step_fields'])!=set(STEP_FIELDS): raise ValueError('incomplete full oracle')
    require_exact([s['pass'] for s in record['steps']],record['summary']['sequence'],'sequence')
    if len(record['steps'])!=len(record['joint_trace']): raise ValueError('joint trace length')
    rows=[]; previous=q_home
    for s,j in zip(record['steps'],record['joint_trace']):
        require_exact(j['q_start'],previous,'joint continuity')
        require_exact(j['q_end'],j['q_samples'][-1],'joint scan end')
        rows.append(dict(kind='scan',pass_index=s['pass'],t_move_s=s['t_move'],t_scan_s=s['t_scan'],t_home_s=0.,
            cumulative_time_s=j['elapsed_before_home'],primary_coverage=s['coverage'],
            joint_start_rad=j['q_start'],joint_samples_rad=j['q_samples'],joint_end_rad=j['q_end'],
            repair=s['repair'],original_step=s))
        previous=j['q_end']
    last=record['steps'][-1]
    rows.append(dict(kind='home',pass_index=None,t_move_s=0.,t_scan_s=0.,t_home_s=last['t_home'],
        cumulative_time_s=record['summary']['cycle_time'],primary_coverage=last['coverage'],
        joint_start_rad=previous,joint_end_rad=q_home,repair=False))
    return dict(method=record['method'],seed=record['seed'],source_run_id=record['source_run'],
        variant=record['variant'],summary=record['summary'],steps=rows,
        verification=verify|{'full_oracle_validation_manifest_sha256':manifest_sha,
            'joint_provenance':'Derived by unchanged ENV v4 transfer/wrap/limit routine, matched to returned observations; home uses direct q_home difference as ENV v4. No independently recorded historical learner joint trace.'})


def schema():
    array = {
        'type': 'object',
        'required': ['dtype', 'shape', 'byte_order', 'order', 'encoding', 'data', 'units'],
        'properties': {
            'dtype': {'enum': ['<f8', '<u4', '|u1']},
            'shape': {'type': 'array', 'items': {'type': 'integer', 'minimum': 0}},
            'byte_order': {'enum': ['little', 'not-applicable']},
            'order': {'const': 'C'}, 'encoding': {'const': 'base64'},
            'data': {'type': 'string', 'contentEncoding': 'base64'},
            'units': {'type': 'string'},
        },
    }
    typed = {'$ref': '#/$defs/array'}
    verification = {
        'type': 'object',
        'required': ['stage', 'exact', 'matched_fields', 'final_coverage_status'],
        'properties': {'stage': {'enum': ['A', 'B']}, 'exact': {'const': True}},
    }
    episode = {
        'type': 'object',
        'required': ['method', 'seed', 'source_run_id', 'variant', 'summary', 'steps', 'verification'],
        'properties': {
            'method': {'enum': list(METHODS)}, 'seed': {'const': 0},
            'steps': {'type': 'array', 'minItems': 2}, 'verification': verification,
        },
    }
    pass_keys = ['sensor_start_mm', 'sensor_end_mm', 'view', 'sweep', 'q_samples_rad', 'valid_q', 'mask_reason']
    part_keys = ['part', 'variant', 'mesh', 'points', 'normals', 'measured_bits', 'passes', 'episodes', 'attribution']
    part_properties = {
        'kind': {'const': 'part'}, 'part': {'enum': list(VAL_PARTS)}, 'variant': {'type': 'string'},
        'mesh': {'type': 'object', 'required': ['vertices', 'triangles', 'simplification'],
                 'properties': {'vertices': typed, 'triangles': typed}},
        'points': typed, 'normals': typed, 'measured_bits': typed,
        'passes': {'type': 'object', 'required': pass_keys, 'properties': {key: typed for key in pass_keys}},
        'episodes': {'type': 'array', 'minItems': 7, 'maxItems': 7, 'items': episode},
        'attribution': {
            'type': 'object',
            'required': ['name', 'author', 'source_url', 'license', 'license_version',
                         'official_specific_license_url', 'version_status', 'changes', 'share_alike'],
        },
    }
    return {
        '$schema': 'https://json-schema.org/draft/2020-12/schema',
        'title': 'NBV Plan 004 VAL demo bundle v1', '$defs': {'array': array}, 'type': 'object',
        'required': ['schema_version', 'kind', 'coordinates', 'provenance'],
        'properties': {
            'schema_version': {'const': 1}, 'kind': {'enum': ['common', 'part']},
            'coordinates': {'type': 'object'}, 'provenance': {'type': 'object'},
        },
        'oneOf': [
            {'properties': {'kind': {'const': 'common'}, 'robot': {'type': 'object'}, 'styles': {'type': 'object'}},
             'required': ['robot', 'styles']},
            {'properties': part_properties, 'required': part_keys},
        ],
    }


def validate_document(document):
    Draft202012Validator(schema()).validate(document)
    json.dumps(document,allow_nan=False)  # catches invalid scalar values outside typed arrays
    def visit(item):
        if isinstance(item,dict):
            if 'dtype' in item and 'data' in item: decode_array(item)
            else:
                for value in item.values(): visit(value)
        elif isinstance(item,list):
            for value in item: visit(value)
    visit(document)
    if document['kind']=='part':
        if document['variant']!=f'{document["part"]}_v0': raise ValueError('variant identity')
        if {r['method'] for r in document['episodes']}!=set(METHODS): raise ValueError('seven distinct methods')
        if document['points']['shape']!=[20000,3] or document['normals']['shape']!=[20000,3]: raise ValueError('sample correspondence shape')
        bits=document['measured_bits']
        if bits['logical_shape']!=[48,20000] or bits['bit_count']!=960000 or bits['shape']!=[48,2500] or bits['bit_order']!='big': raise ValueError('visibility shape/order')
        p=document['passes']
        for key,shape in [('sensor_start_mm',[48,3]),('sensor_end_mm',[48,3]),('view',[48,3]),('sweep',[48,3]),('q_samples_rad',[48,5,6]),('valid_q',[48]),('mask_reason',[48])]:
            if p[key]['shape']!=shape: raise ValueError(f'pass array shape: {key}')
        valid=decode_array(p['valid_q']).astype(bool); reasons=decode_array(p['mask_reason']);q=decode_array(p['q_samples_rad'])
        if not np.array_equal(valid,reasons==0) or np.any(q[~valid]!=0): raise ValueError('invalid pass animation mask')
        vertices=decode_array(document['mesh']['vertices']);triangles=decode_array(document['mesh']['triangles'])
        if len(triangles)>50000 or triangles.ndim!=2 or triangles.shape[1]!=3 or triangles.max()>=len(vertices): raise ValueError('display mesh')
        for episode in document['episodes']:
            if episode['variant']!=document['variant']: raise ValueError('episode identity')
            if not all(valid[s['pass_index']] for s in episode['steps'] if s['kind']=='scan'): raise ValueError('episode animates invalid pass')
    return document


def part_document(fid,records,provenance,manifest_sha):
    if fid not in VAL_PARTS: raise ValueError('Only three approved VAL v0 parts may be exported')
    scene=assets.load_scene(fid,variant=True)
    plan=scene['plan']; valid=plan['reachable'].astype(bool);ik=plan['ik_reachable'].astype(bool)
    q=plan['q'].copy();q[~valid]=0.  # explicitly declared sentinel, never animated
    midpoint=plan['center']+plan['standoff'][:,None]*plan['direction']
    half=plan['length'][:,None]*plan['sweep']/2
    selected=[r for r in records if r['part']==fid and r['variant']==f'{fid}_v0' and r['seed']==0 and r['method'] in METHODS]
    if len(selected)!=7: raise ValueError('Seven verified seed-0 episodes required')
    robot=assets.robot_description()
    document=dict(schema_version=1,kind='part',part=fid,variant=f'{fid}_v0',coordinates=COORDINATES,
        provenance=provenance|{'source_files_sha256':scene['source_files']},attribution=attribution(scene['attribution']),
        mesh={'vertices':encode_array(scene['mesh_vertices'],'mm'),'triangles':encode_array(scene['mesh_triangles'],'vertex index'),
              'simplification':{'applied':False,'triangle_limit':50000,'parameters':None},'geometry':'normalized nominal; not variant geometry'},
        points=encode_array(scene['variant_points'],'mm'),normals=encode_array(scene['variant_normals'],'unit vector'),
        measured_bits=pack_visibility(scene['variant_measured']),
        passes=dict(sensor_start_mm=encode_array(midpoint-half,'mm'),sensor_end_mm=encode_array(midpoint+half,'mm'),
            view=encode_array(plan['view'],'unit vector'),sweep=encode_array(plan['sweep'],'unit vector'),
            q_samples_rad=encode_array(q,'rad'),valid_q=encode_array(valid.astype('u1'),'boolean'),
            mask_reason=encode_array(np.where(valid,0,np.where(ik,2,1)).astype('u1'),'enum'),
            mask_legend={'0':'reachable','1':'IK unreachable','2':'collision masked'},
            invalid_q='Zero-filled only where valid_q=false. Do not animate these passes.',
            geometry_provenance='Saved nominal RobotPlan geometry and five IK samples. No IK or new visibility computation.',
            joint_convention='Stored nominal IK angles; executed lifted angles are in episode steps (ENV v4 wrapping/limits).'),
        episodes=[episode_document(next(r for r in selected if r['method']==method),robot['q_home_rad'],manifest_sha) for method in METHODS])
    return validate_document(document)


def write_manifest(directory,*,commit):
    directory=Path(directory)
    files={p.name:{'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(directory.iterdir()) if p.is_file() and p.name!='manifest.json'}
    item=dict(schema_version=1,commit=commit,files=files,hash_scope='All other bundle files; manifest cannot contain its own SHA-256. External export report records manifest SHA-256.',
        budget_bytes=12000000,payload_bytes=sum(e['bytes'] for e in files.values()))
    (directory/'manifest.json').write_text(json.dumps(item,indent=2),encoding='utf-8')


def check_bundle(directory,limit=12000000):
    directory=Path(directory);manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    actual={p.name for p in directory.iterdir() if p.is_file()}
    if actual!=set(manifest['files'])|{'manifest.json'}: raise ValueError('bundle file list mismatch')
    for name,entry in manifest['files'].items():
        path=directory/name
        if digest(path)!=entry['sha256'] or path.stat().st_size!=entry['bytes']: raise ValueError(f'bundle hash mismatch: {name}')
    total=sum((directory/name).stat().st_size for name in actual)
    if total>limit: raise ValueError('bundle exceeds 12000000-byte budget (manifest included)')
    return dict(total_bytes=total,files={name:(directory/name).stat().st_size for name in sorted(actual)},manifest_sha256=digest(directory/'manifest.json'))


def publish_bundle(stage,target):
    """Retain verified bytes, inherit destination ACLs (Windows mkdtemp is private)."""
    stage,target=Path(stage),Path(target)
    before=check_bundle(stage)
    target.mkdir(exist_ok=False)
    # Fresh files inherit results/ access; copy neither directory ACLs nor file metadata.
    for name in sorted(before['files'],key=lambda name:(name=='manifest.json',name)):
        with (target/name).open('xb') as stream: stream.write((stage/name).read_bytes())
    if check_bundle(target)!=before: raise ValueError('published bundle differs from verified scratch')


VIEWER_NOTES='''# NBV VAL demo bundle v1

Upload all seven files together: common.json, part_271875_v0.json, part_591247_v0.json,
part_37275_v0.json, schema.json, manifest.json and README.md. No test data is present.
Use schema.json and the semantic rules below; verify SHA-256 before displaying.

Arrays: base64 raw C-order bytes; <f8 = little-endian IEEE-754 binary64,
<u4 = little-endian uint32, |u1 = uint8 (byte order not applicable).
No lossy rounding/downcasting. Positions are mm, angles rad, time s.
Visibility is 48 pass-major rows, each 2,500 bytes. Big bit order: sample 0 is
the most significant bit of byte 0; exactly 20,000 bits per row, no padding.
Sample positions AND normals are variant-v0 values in their original correspondence
order. Display mesh is normalized nominal, not variant. Do not reorder samples.
Mesh nominal, points variant v0; deviations up to approximately 1.5 mm.

Robot uses standard DH: RotZ(theta+offset) TransZ(d) TransX(a) RotX(alpha),
with metres in DH, rad in joint states. Apply base yaw/translation, then flange TCP
translation along local z. Part/world frame: bbox bottom-center, z up. Capsule
coordinates are mm and DH frame indices; offsets are along joint-2 axis as in
configs/collision_v1.yaml. Sensor box is in sensor frame (x=sweep, z=view).
Never animate valid_q=0; those q entries contain declared zero placeholders.
The five nominal IK samples may differ by 2*pi lifts from an episode's executed
samples. Use episode joint_samples_rad, which preserve ENV v4 wrapping/limits.
Move starts at joint_start_rad, reaches the first sample, then scans through five
samples; no physical intermediate trajectory is certified. Scan-step cumulative
time excludes home. The final home row uses direct q_home return as ENV v4.
original_step retains the exact evaluation record, whose final cycle_time already
includes home. Avoid adding that home duration twice.

Episodes are stored-sequence replays, no policy calls. Full oracle A: 195 canonical
VAL baseline episodes, all 12 original step fields plus summaries exactly matched.
Partial oracle B: 900 learner VAL episodes, sequence and all 15 stored summary fields
exactly matched. Every exported episode carries its verification and manifest link.
Learner final coverage: derived from replay; consistent with partial oracle.
JSON reference decimals are parsed back to IEEE-754 binary64 and compared by bytes;
no tolerances or rounded display-cell comparisons. Historical learner joints/final
coverage were not independently stored. AUC is the step-function integral of primary
coverage on [0,180 s] / 180, with jumps at completed passes and constant home coverage;
matching AUC is one scalar consistency check, not a proof of final coverage.

Always show each part's name, author, Thingiverse/Thingi10K source link, recorded
CC BY / CC BY-SA family, version status and changes alongside its mesh/derivatives.
All three license versions are NOT recorded in the split metadata: do not infer 4.0.
official_specific_license_url is null; the CC overview link is only an overview.
BY-SA derivatives retain the same license; attribution, license link and changed-
material notice are required. Exact version-specific license links require resolving
the missing source metadata before public distribution. Transformations are listed
per component in attribution.changes; no display simplification was performed.
'''


def export(directory='results/demo'):
    state=git_state()
    if state['dirty']: raise ValueError('commit exporter before producing bundle')
    target=Path(directory).resolve();workspace=Path.cwd().resolve()
    if not target.is_relative_to(workspace/'results') or target.exists(): raise ValueError('new output directory under results/ required')
    records,validation=validated_records();manifest_sha=digest('results/visualization/replay_validation_v1.json')
    provenance=dict(commit=state['commit'],replay_commit=validation['commit'],replay_manifest_sha256=manifest_sha,
        registry_sha256=digest('results/registry.csv'),source_run_ids=sorted({r['source_run'] for r in records if r['seed']==0 and r['method'] in METHODS}),split='val',no_policy_calls=True,no_test_access=True)
    robot=assets.robot_description()
    common=validate_document(dict(schema_version=1,kind='common',coordinates=COORDINATES,provenance=provenance,robot=robot,styles=STYLES))
    documents={'common.json':common,'schema.json':schema()}
    for fid in VAL_PARTS: documents[f'part_{fid}_v0.json']=part_document(fid,records,provenance,manifest_sha)
    scratch=workspace/'scratch';scratch.mkdir(exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='p004-demo-',dir=scratch)).resolve()
    for name,item in documents.items():
        (stage/name).write_text(json.dumps(item,ensure_ascii=False,allow_nan=False,separators=(',',':')),encoding='utf-8')
    (stage/'README.md').write_text(VIEWER_NOTES,encoding='utf-8')
    write_manifest(stage,commit=state['commit']);result=check_bundle(stage)
    # Publish only a wholly verified, within-budget bundle. No existing directory is replaced.
    if not stage.is_relative_to(workspace/'scratch') or not target.is_relative_to(workspace/'results'): raise ValueError('output boundary')
    publish_bundle(stage,target)
    result.update(commit=state['commit'],directory=target.relative_to(workspace).as_posix(),episodes=21,parts=list(VAL_PARTS),
        missing_license_versions=list(VAL_PARTS),verification=validation['counts'],coordinate_warning=COORDINATES['warning'])
    Path('results/visualization/demo_export_v1.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result
