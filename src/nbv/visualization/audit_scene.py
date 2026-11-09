"""Local snapshots of D-071 worst recorded paths; allowed geometry only, no policies."""
import json
from pathlib import Path
import numpy as np
from spatialmath import SE3
from nbv.collision_audit import allowed_part
from nbv.visualization.assets import load_scene,robot_description
from nbv.visualization.live_scene import Scene


def worst_events(events,count=3):
    selected=[];seen=set()
    for e in sorted(events,key=lambda e:-e['penetration_mm']):
        allowed_part(e['part'])
        if e['geometry_id'] in seen:continue
        selected.append(e);seen.add(e['geometry_id'])
        if len(selected)==count:break
    return selected


def create_audit_server(port=8081,path='results/collision_audit_v1.json'):
    import viser
    report=json.loads(Path(path).read_text(encoding='utf-8'))
    if not report['complete']:raise ValueError('incomplete audit cannot be presented')
    events=worst_events(report['events'])
    server=viser.ViserServer(host='127.0.0.1',port=port,label='D-071 collision diagnosis')
    server.scene.set_up_direction('+z');server.gui.configure_theme(show_share_button=False)
    server.gui.add_markdown('# D-071 — collision diagnosis\nRecorded train/VAL paths; no correction, policy or test access.')
    selector=server.gui.add_dropdown('Event',[str(i+1) for i in range(len(events))],initial_value='1')
    status=server.gui.add_markdown('');current=[None]
    def render():
        e=events[int(selector.value)-1];g=report['geometries'][e['geometry_id']];worst=g['worst']
        if current[0]:current[0].root.remove()
        data=load_scene(e['part'],variant=True,variant_index=int(e['variant'].rsplit('_v',1)[1]))
        scene=Scene(server,data,robot_description());current[0]=scene
        q=np.asarray(worst['worst_q_rad']);scene.draw_q(q)
        if e['body'] in scene.capsules:scene.capsules[e['body']].color=(213,30,30);scene.capsules[e['body']].opacity=1.
        else:
            T=scene.robot.fkine(q).A;lo,hi=scene.geom.box_lo/1000,scene.geom.box_hi/1000
            from scipy.spatial.transform import Rotation
            scene.api.add_box('/nbv/sensor',dimensions=hi-lo,position=T[:3,3]+T[:3,:3]@((lo+hi)/2),
                wxyz=Rotation.from_matrix(T[:3,:3]).as_quat(scalar_first=True),color=(213,30,30))
        status.content=(f"**{e['variant']} · {e['method']} · seed {e['seed']}**\n\n"
            f"{e['phase']} · step {e['step']} · {e['body']} / {e['obstacle']}\n\n"
            f"Penetration **{e['penetration_mm']:.6f} mm** · 1° sampled path\n\n"
            f"{e['source_run']}\n\nNominal D-042 collision geometry; variant points are display context. "
            'No self-collision or fixture model. Red = implicated collision body.')
        server._nbv_audit_event=e
    selector.on_update(lambda _:render())
    @server.on_client_connect
    def camera(client):
        client.camera.up_direction=(0,0,1);client.camera.position=(.65,-.85,.60);client.camera.look_at=(-.12,0,.16)
    render();return server
