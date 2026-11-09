"""Local presentation helpers; no policy, environment, test input or result writes."""
import functools
import hashlib
import html
import json
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from nbv.visualization.styles import styles_for

METHODS=('greedy_cost','mcts','ilp_tsp','joint_opt','sarsa/linear','sarsa/value_estimate','regq/linear')
COLORS=styles_for(METHODS)


class ScopedScene:
    def __init__(self,api,prefix):self.api=api;self.prefix=prefix
    def __getattr__(self,name):
        method=getattr(self.api,name)
        if not name.startswith('add_'):return method
        def scoped(path,*args,**kwargs):return method(self.prefix+path.removeprefix('/nbv'),*args,**kwargs)
        return scoped


def lights(server):
    server.scene.configure_environment_map(None)
    server.scene.configure_default_lights(False)
    server.scene.add_light_ambient('/presentation/ambient',intensity=.5)
    server.scene.add_light_directional('/presentation/key',intensity=2.3,cast_shadow=True,
        wxyz=Rotation.from_euler('xyz',[-.65,.45,.2]).as_quat(scalar_first=True))
    server.scene.add_light_directional('/presentation/fill',intensity=.7,
        wxyz=Rotation.from_euler('xyz',[.9,-.7,1.5]).as_quat(scalar_first=True))


def camera(server,name,*,comparison=False):
    y=.45 if comparison else 0.
    poses={'Genel':((.95,-1.05+y,.85),(-.12,y,.16)),
           'Parça':((.4,-.45+y,.38),(0,y,.12)),
           'Yan':((.8,y,.28),(-.15,y,.18))}
    position,target=poses[name]
    for client in server.get_clients().values():
        client.camera.up_direction=(0,0,1);client.camera.position=position;client.camera.look_at=target


def coverage_html(coverage):
    value=float(np.clip(coverage,0,1))*100
    return (f'<div style="font:13px sans-serif;padding:8px 0">Kapsama <b>{value:.2f}%</b>'
        '<div style="position:relative;height:15px;background:#dfe5eb;border-radius:4px;margin:8px 0 20px">'
        f'<div style="width:{value:.8f}%;height:100%;background:#0072B2;border-radius:4px"></div>'
        '<div style="position:absolute;left:95%;top:-4px;height:23px;border-left:2px solid #222"></div>'
        '<span style="position:absolute;right:0;top:19px;font-size:11px">hedef %95</span></div></div>')


def load_audit():
    from nbv.visualization.playback import AuditIndex
    for name in ('results/collision_audit_v1.json','results/collision_audit_val_v1.json'):
        path=Path(name)
        if path.exists():return AuditIndex(json.loads(path.read_text(encoding='utf-8')))
    return None


@functools.lru_cache(maxsize=16)
def live_reference_curves(variant,seed):
    from nbv.visualization.live import allowed_variant
    from nbv.visualization.replay import validated_records,digest
    from nbv.visualization.demo import episode_document
    from nbv.visualization.assets import robot_description
    fid,index=variant.rsplit('_v',1);allowed_variant(int(fid),int(index))
    records,_=validated_records();chosen={}
    for e in records:
        expected=0 if e['method'] in ('greedy_cost','ilp_tsp','joint_opt') else seed
        if e['variant']==variant and e['method'] in METHODS and e['seed']==expected:chosen[e['method']]=e
    sha=digest('results/visualization/replay_validation_v1.json');home=tuple(float(q) for q in robot_description()['q_home_rad'])
    return [episode_document(chosen[m],home,sha) for m in METHODS]


@functools.lru_cache(maxsize=14)
def mesh_glb(path):
    import trimesh
    return trimesh.load(path,force='scene').export(file_type='glb')


class UR5Visual:
    def __init__(self,api,base):
        from nbv.visualization.ur5_visual import UR5VisualModel
        self.model=UR5VisualModel();self.base=base;self.handles=[]
        path=Path('results/visualization/urdf_alignment_v1.json')
        if not path.exists():raise ValueError('saved-pose URDF/DH validation not yet available; capsules retained')
        validation=json.loads(path.read_text(encoding='utf-8'))
        sha=hashlib.sha256((self.model.directory/'manifest.json').read_bytes()).hexdigest()
        if not validation['passed'] or validation['asset_manifest_sha256']!=sha:
            raise ValueError('URDF/DH validation failed or asset identity changed; capsules retained')
        print('UR5 visual assets: BSD-3-Clause; example-robot-data '+self.model.manifest['source_commit']+'; offline, separate from D-042 collision geometry.',flush=True)
        for name,path,origin in self.model.visuals():
            handle=api.add_glb(f'/nbv/ur5/{name}',mesh_glb(str(path)),cast_shadow=True,receive_shadow=True)
            self.handles.append((name,origin,handle))

    def draw(self,q):
        frames=self.model.frames(q)
        for name,origin,handle in self.handles:
            T=self.base@self.model.base_alignment@frames[name]@origin
            handle.position=T[:3,3];handle.wxyz=Rotation.from_matrix(T[:3,:3]).as_quat(scalar_first=True)


def attribution_text(a):
    return (f"**{a['name']}** — {a['author']}\n\n[Thingi10K / kaynak]({a['source_url']}) · "
        f"{a['license']} · **sürüm kayıtlı değil**. [CC genel bakış]({a['official_license_overview']}).\n\n"
        'Değişiklikler: 200 mm ölçekleme, alan ağırlıklı örnekler, varyant sapmaları ve türetilmiş görünürlük. '
        'BY-SA türevleri aynı lisans ailesini korur.\n\n'
        '**Robot:** example-robot-data / ROS-Industrial UR5; BSD-3-Clause. Ağlar görsel modeldir; '
        'D-042 kapsülleri ve sensör kutusu çarpışma modelidir. Kaynak lisansı assets/ur5/LICENSE içinde korunur.')


def brief_credit(a):
    return ('<div style="font:11px sans-serif;color:#555;padding:4px 0">'
        f"Part: {html.escape(a['name'])} — {html.escape(a['author'])}; {html.escape(a['license'])} "
        f"(sürüm kayıtlı değil). <a href=\"{html.escape(a['source_url'],quote=True)}\" target=\"_blank\" rel=\"noopener noreferrer\">Kaynak</a>"
        '<br>Robot: UR5 / example-robot-data, BSD-3-Clause. Ayrıntı ve değişiklikler: Bilgi.</div>')


class CoverageChart:
    def __init__(self,gui):
        import plotly.graph_objects as go
        self.handle=gui.add_plotly(go.Figure(),aspect=1.5,config={'displayModeBar':False});self.episodes=[];self.chosen=None;self.last=0.

    def reset(self,episodes,chosen):self.episodes=episodes;self.chosen=chosen;self.update(0.,force=True)

    def update(self,time_s,*,actual_rows=None,force=False):
        if not force and time.monotonic()-self.last<.25:return
        self.last=time.monotonic()
        import plotly.graph_objects as go
        fig=go.Figure()
        for episode in self.episodes:
            selected=episode['method']==self.chosen
            rows=actual_rows if selected and actual_rows is not None else episode['steps']
            x=[0.]+[r['cumulative_time_s'] for r in rows];y=[0.]+[r['primary_coverage'] for r in rows]
            if selected and actual_rows is not None and time_s>x[-1]:x.append(time_s);y.append(y[-1])
            fig.add_trace(go.Scatter(x=x,y=y,name=episode['method'],mode='lines',line=dict(shape='hv',
                color=COLORS[episode['method']]['color'] if selected else 'rgba(110,120,130,0.25)',width=3 if selected else 1)))
        fig.add_hline(y=.95,line_dash='dash',line_color='#777')
        fig.add_vline(x=time_s,line_color='#222',line_width=1)
        maximum=max((e['steps'][-1]['cumulative_time_s'] for e in self.episodes),default=1.)
        fig.update_layout(margin=dict(l=32,r=8,t=10,b=32),showlegend=False,height=250,
            paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(size=11),
            xaxis=dict(title='Birikmiş süre (s)',range=[0,maximum]),yaxis=dict(title='Kapsama',range=[0,1.03],tickformat='.0%'))
        self.handle.figure=fig
