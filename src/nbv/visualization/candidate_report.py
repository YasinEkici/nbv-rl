"""Read-only candidate audit, strictly train/VAL nominal caches."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from nbv import parts
from nbv.visualization.assets import require_scene_part
from nbv.visualization.replay import digest


def audit():
    manifest=parts.load_yaml('configs/split_v1.yaml');records=[];plans=[]
    for p in manifest['parts']:
        if p['split'] not in ('train','val'):continue
        fid=p['file_id'];require_scene_part(fid,manifest)
        path=Path(f'data/cache/robot/nominal/{fid}_n48.npz')
        with np.load(path) as z: plan={key:z[key] for key in z.files}
        valid=plan['reachable'].astype(bool);ik=plan['ik_reachable'].astype(bool)
        reason=np.where(valid,0,np.where(ik,2,1)).astype('u1')
        midpoint=plan['center']+plan['standoff'][:,None]*plan['direction']
        start=midpoint-plan['length'][:,None]*plan['sweep']/2
        end=midpoint+plan['length'][:,None]*plan['sweep']/2
        plans.append((reason,plan['direction'],start,end))
        records.append(dict(part=fid,split=p['split'],reachable=int(valid.sum()),ik_unreachable=int((~ik).sum()),
            collision_masked=int((ik&~valid).sum()),mask_reason=reason.tolist(),
            direction_sha256=hashlib.sha256(plan['direction'].tobytes()).hexdigest(),
            start_sha256=hashlib.sha256(start.tobytes()).hexdigest(),end_sha256=hashlib.sha256(end.tobytes()).hexdigest(),
            source=path.as_posix(),sha256=digest(path)))
    first=plans[0]
    result=dict(scope='Only the eleven train/VAL base parts; the three test caches were not opened. Equality across all fourteen is NOT established.',
        parts=records,all_allowed_mask_reason_equal=all(np.array_equal(p[0],first[0]) for p in plans),
        all_allowed_directions_equal=all(np.array_equal(p[1],first[1]) for p in plans),
        all_allowed_sensor_positions_equal=all(np.array_equal(p[2],first[2]) and np.array_equal(p[3],first[3]) for p in plans),
        explanation='Directions are part-independent Fibonacci-cap vectors (candidates.py). Center, support/standoff and sweep length depend on the nominal mesh. robot.plan_passes solves each part separately: IK/table/step constraints, then min(robot-part, robot-table) clearance at five samples. No constant mask is hard-coded. Equal masks are an empirical coincidence in this fixed cell/normalized size, not proof that part geometry is irrelevant. Combined cached min_clearance alone does not identify the limiting body or establish table-only causation.',
        test_geometry_access=False)
    # Existing first-VAL diagnostic prefix only; never traverse later/test-part rows.
    evidence=[]
    with Path('results/collision_v2.csv').open(encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            if int(row['base'])!=records[0]['part']:break
            if row['reachable_v4']=='False':evidence.append(row)
    result['existing_first_val_collision_evidence']=evidence
    out=Path('results/visualization');out.mkdir(exist_ok=True)
    (out/'candidate_audit_v1.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    lines=['# Candidate-mask audit (A3)','',result['scope'],'',
        '| Part | Split | Reachable | IK failure | Collision masked |','|---|---|---:|---:|---:|']
    for r in records:lines.append(f'| {r["part"]} | {r["split"]} | {r["reachable"]} | {r["ik_unreachable"]} | {r["collision_masked"]} |')
    lines+=['',f'Identical allowed mask_reason arrays: **{result["all_allowed_mask_reason_equal"]}**. Identical directions: **{result["all_allowed_directions_equal"]}**. Identical sensor start/end positions: **{result["all_allowed_sensor_positions_equal"]}**.','',result['explanation'],'',
        'Source path and raw SHA-256 per allowed cache, mask arrays and geometry hashes: [JSON](candidate_audit_v1.json). No new IK, ray casting or evaluation.','',
        '## Existing first-VAL clearance evidence','',
        'The following rounded diagnostics were already recorded in results/collision_v2.csv. Only its first VAL-part prefix was read; no later test rows. These are supporting diagnostics, not a fresh collision computation.','',
        '| Pass | Part clearance mm / body | Table clearance mm / body |','|---:|---|---|']
    for e in evidence:lines.append(f'| {e["pass"]} | {e["part_clearance_mm"]} / {e["part_body"]} | {e["table_clearance_mm"]} / {e["table_body"]} |')
    lines+=['','Both part and table violations occur in this saved example; do not attribute every masked direction to the table. IK failure does not identify a single cause either. For the other parts the combined cache alone is insufficient to separate causes; we are uncertain rather than claiming a causal diagnostic.','']
    (out/'candidate_audit_v1.md').write_text('\n'.join(lines),encoding='utf-8')
    return result
