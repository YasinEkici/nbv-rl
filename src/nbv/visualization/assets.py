"""Explicit train/VAL geometry reads; the test split is rejected before opening any asset."""
import importlib.metadata
from pathlib import Path
import numpy as np
from nbv import parts,robot,visibility
from nbv.visualization.replay import digest


def require_scene_part(fid,manifest):
    matches=[p for p in manifest['parts'] if str(p['file_id'])==str(fid)]
    if len(matches)!=1 or matches[0]['split'] not in ('train','val'): raise ValueError('closed test holdout: geometry forbidden')
    return matches[0]['split']


def load_scene(fid,*,variant=False):
    fid=int(fid); manifest=parts.load_yaml('configs/split_v1.yaml'); split=require_scene_part(fid,manifest)
    p=next(p for p in manifest['parts'] if p['file_id']==fid)
    point_path=Path(f'data/cache/visibility/nominal/points_{fid}.npz')
    robot_path=Path(f'data/cache/robot/nominal/{fid}_n48.npz')
    nominal_path=Path(f'data/cache/visibility/nominal/{fid}_n48.npz')
    with np.load(point_path) as z: points,normals=z['points'],z['normals']
    with np.load(robot_path) as z: plan={k:z[k] for k in z.files}
    measured=visibility.Visibility.load(nominal_path).measured()
    paths=[point_path,robot_path,nominal_path]
    scene=dict(part=fid,split=split,points=points,normals=normals,nominal_measured=measured,plan=plan,attribution=p)
    if variant:
        import trimesh
        cfg=parts.load_yaml('configs/parts_v2.yaml')
        raw=Path(cfg['paths']['raw'])/f'{fid}.ply'
        variant_path=Path(cfg['paths']['variants'])/f'{fid}_v0.ply'
        samples_path=Path(f'data/cache/visibility/samples_{fid}.npz')
        variant_vis=Path(f'data/cache/visibility/{fid}_v0_n48.npz')
        mesh=parts.normalize(trimesh.load(raw,process=False),cfg['geometry']['longest_mm'])
        variant_mesh=trimesh.load(variant_path,process=False)
        with np.load(samples_path) as z: sample=visibility.SurfaceSamples(z['face'],z['bary'])
        positions,normal=sample.on(variant_mesh)
        scene.update(mesh_vertices=np.asarray(mesh.vertices),mesh_triangles=np.asarray(mesh.faces,dtype='<u4'),
            variant_points=positions,variant_normals=normal,variant_measured=visibility.Visibility.load(variant_vis).measured())
        paths += [raw,variant_path,samples_path,variant_vis]
    scene['source_files']={path.as_posix():digest(path) for path in paths}
    return scene


def robot_description():
    cfg=parts.load_yaml('configs/robot_v2.yaml'); model=robot.make_robot(cfg)
    return dict(model='UR5',source='nbv.robot.make_robot / roboticstoolbox DH UR5',
        source_version=importlib.metadata.version('roboticstoolbox-python'),
        dh={'a_m':[float(link.a) for link in model.links],'d_m':[float(link.d) for link in model.links],
            'alpha_rad':[float(link.alpha) for link in model.links],'offset_rad':[float(link.offset) for link in model.links]},
        base_xyz_mm=cfg['base_xyz_mm'],base_yaw_deg=cfg['base_yaw_deg'],tcp_offset_mm=cfg['tcp_offset_mm'],
        q_home_rad=robot.q_home(cfg).tolist(),joint_limit_deg=cfg['joint_limit_deg'],
        collision=parts.load_yaml('configs/collision_v1.yaml'))
