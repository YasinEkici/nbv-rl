"""Offline BSD-3-Clause UR5 visuals, independently checked against the frozen DH model."""
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

JOINTS=('shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint')


def origin(node):
    T=np.eye(4)
    if node is not None:
        T[:3,3]=np.fromstring(node.get('xyz','0 0 0'),sep=' ')
        T[:3,:3]=Rotation.from_euler('xyz',np.fromstring(node.get('rpy','0 0 0'),sep=' ')).as_matrix()
    return T


class UR5VisualModel:
    def __init__(self,directory='assets/ur5'):
        self.directory=Path(directory)
        path=self.directory/'manifest.json'
        if not path.exists():raise ValueError('Offline UR5 assets missing; restore assets/ur5. No download is attempted.')
        self.manifest=json.loads(path.read_text(encoding='utf-8'))
        for rel,sha in self.manifest['files'].items():
            target=(self.directory/rel).resolve()
            if not target.is_relative_to(self.directory.resolve()) or not target.is_file():raise ValueError('Offline UR5 asset missing')
            if hashlib.sha256(target.read_bytes()).hexdigest()!=sha:raise ValueError(f'UR5 asset hash mismatch: {rel}')
        self.root=ET.fromstring((self.directory/'model.urdf').read_text(encoding='utf-8'))
        self.joints=[];self.joint_offsets=np.zeros(6)
        for j in self.root.findall('joint'):
            axis=np.fromstring(j.find('axis').get('xyz','1 0 0'),sep=' ') if j.find('axis') is not None else np.array([1.,0.,0.])
            self.joints.append(dict(name=j.get('name'),kind=j.get('type'),parent=j.find('parent').get('link'),
                child=j.find('child').get('link'),origin=origin(j.find('origin')),axis=axis))
        children={j['child'] for j in self.joints};roots={l.get('name') for l in self.root.findall('link')}-children
        if len(roots)!=1:raise ValueError('URDF must have exactly one root')
        self.root_name=roots.pop()
        self.base_alignment=np.eye(4);self.base_alignment[:3,:3]=Rotation.from_rotvec([0,0,np.pi]).as_matrix()

    def frames(self,q):
        q=np.asarray(q,dtype=float)
        if q.shape!=(6,) or not np.isfinite(q).all():raise ValueError('finite six-joint UR5 pose required')
        values=dict(zip(JOINTS,q+self.joint_offsets));frames={self.root_name:np.eye(4)};pending=list(self.joints)
        while pending:
            ready=[j for j in pending if j['parent'] in frames]
            if not ready:raise ValueError('URDF graph is disconnected or cyclic')
            for j in ready:
                R=np.eye(4)
                if j['kind'] in ('revolute','continuous'):
                    R[:3,:3]=Rotation.from_rotvec(j['axis']*values[j['name']]).as_matrix()
                elif j['kind']!='fixed':raise ValueError('unsupported visual URDF joint')
                frames[j['child']]=frames[j['parent']]@j['origin']@R;pending.remove(j)
        return frames

    def tool_pose(self,q,*,base=None,tcp_offset_m=0.):
        tool=np.eye(4);tool[2,3]=tcp_offset_m
        return (np.eye(4) if base is None else base)@self.base_alignment@self.frames(q)['tool0']@tool

    def visuals(self):
        """Original visual origins/units; package URIs resolve only to vendored files."""
        for link in self.root.findall('link'):
            for visual in link.findall('visual'):
                mesh=visual.find('geometry/mesh')
                if mesh is None:continue
                rel=mesh.get('filename').split('/meshes/',1)[1]
                path=self.directory/'meshes'/rel
                if not path.is_file():raise ValueError('Offline UR5 visual mesh missing')
                yield link.get('name'),path,origin(visual.find('origin'))


def verify_saved_poses(destination='results/visualization/urdf_alignment_v1.json'):
    from nbv import parts,robot,evaluation
    from nbv.collision_audit import allowed_part
    model=UR5VisualModel();cfg=parts.load_yaml('configs/robot_v2.yaml');dh=robot.make_robot(cfg)
    rows=[]
    for p in parts.load_yaml('configs/split_v1.yaml')['parts']:
        if p['split'] not in ('train','val'):continue
        fid=p['file_id'];allowed_part(fid)
        with np.load(f'data/cache/robot/nominal/{fid}_n48.npz') as z:
            q=z['q'];valid=np.isfinite(q).all(axis=(1,2));reachable=z['reachable'].astype(bool)
        position=[];angles=[]
        for action in np.flatnonzero(valid):
            for pose in q[action]:
                for offset in (0.,cfg['tcp_offset_mm']/1000):
                    expected=dh.fkine(pose).A@np.linalg.inv(dh.tool.A)
                    expected[:3,3]+=expected[:3,:3]@np.array([0.,0.,offset])
                    actual=model.tool_pose(pose,base=dh.base.A,tcp_offset_m=offset)
                    position.append(float(np.linalg.norm(actual[:3,3]-expected[:3,3])*1000))
                    angles.append(float(np.degrees(np.arccos(np.clip(np.diag(actual[:3,:3].T@expected[:3,:3]),-1,1))).max()))
        rows.append(dict(part=fid,split=p['split'],candidates=len(q),finite_passes=int(valid.sum()),
            reachable_passes=int(reachable.sum()),poses=int(valid.sum()*5),position_mm=max(position),axis_deg=max(angles)))
    passed=all(r['position_mm']<=.5 and r['axis_deg']<=.1 for r in rows)
    report=dict(schema_version=1,passed=passed,parts=rows,poses=sum(r['poses'] for r in rows),
        position_limit_mm=.5,axis_limit_deg=.1,joint_offsets_rad=model.joint_offsets.tolist(),
        base_frame_conversion='URDF base_link to DH base: Rz(pi); no translational fit and no joint offsets.',
        residual='URDF d1=0.089159 m; frozen roboticstoolbox DH d1=0.089459 m. Original dimensions retained.',
        checked='Both flange (tool0) and TCP at every finite saved pass sample, including masked passes. IK-unsolved NaN plans cannot be checked or animated. Train/VAL only.',
        asset_manifest_sha256=hashlib.sha256((model.directory/'manifest.json').read_bytes()).hexdigest(),
        provenance=evaluation.git_state(),license=model.manifest['license'])
    Path(destination).parent.mkdir(parents=True,exist_ok=True)
    Path(destination).write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    return report
