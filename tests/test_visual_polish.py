"""Polished views preserve kinematics, audited collisions and a single comparison clock."""
import json
import numpy as np
import pytest


def test_urdf_flange_and_tcp_match_dh_without_changing_robot_model():
    from nbv.visualization.ur5_visual import UR5VisualModel
    from nbv import parts,robot
    model=UR5VisualModel();cfg=parts.load_yaml('configs/robot_v2.yaml');dh=robot.make_robot(cfg)
    for q in [np.zeros(6),*np.random.default_rng(71).uniform(-np.pi,np.pi,(20,6))]:
        for offset in (0.,cfg['tcp_offset_mm']/1000):
            expected=dh.fkine(q).A@np.linalg.inv(dh.tool.A)
            expected=expected.copy();expected[:3,3]+=expected[:3,:3]@np.array([0,0,offset])
            actual=model.tool_pose(q,base=dh.base.A,tcp_offset_m=offset)
            assert np.linalg.norm(actual[:3,3]-expected[:3,3])*1000<=.5
            cosine=np.clip(np.diag(actual[:3,:3].T@expected[:3,:3]),-1,1)
            assert np.degrees(np.arccos(cosine)).max()<=.1


def test_collision_overlay_uses_all_audited_body_intervals_exactly():
    from nbv.visualization.playback import AuditIndex
    interval={'body':'sensor_head','obstacle':'part','start_fraction':.2,'end_fraction':.6,'penetration_mm':7.25}
    other=interval|{'body':'forearm','penetration_mm':3.}
    event={'id':'run:271875_v0','source_run':'run','variant':'271875_v0','part':271875,
           'phase':'transfer','step':2,'geometry_id':'g','time_s':[3.,5.]}
    index=AuditIndex({'complete':True,'events':[event],'geometries':{'g':{'intervals':[interval,other]}}})
    hits=index.active('run','271875_v0',3.4)
    assert {e['body'] for e in hits}=={'sensor_head','forearm'}
    assert sorted(e['penetration_mm'] for e in hits)==[3.,7.25]
    assert index.active('run','271875_v0',4.21)==[]
    with pytest.raises(ValueError,match='test'):index.active('run','178195_v0',4.)


def _rows(scan,home):
    q=np.zeros((5,6));q[:,0]=np.linspace(0,.4,5)
    return [dict(kind='scan',pass_index=1,t_move_s=1.,t_scan_s=scan,t_home_s=0.,
        cumulative_time_s=1.+scan,primary_coverage=.96,joint_start_rad=np.zeros(6).tolist(),
        joint_samples_rad=q.tolist(),joint_end_rad=q[-1].tolist()),
        dict(kind='home',pass_index=None,t_move_s=0.,t_scan_s=0.,t_home_s=home,
        cumulative_time_s=1.+scan+home,primary_coverage=.96,joint_start_rad=q[-1].tolist(),joint_end_rad=np.zeros(6).tolist())]


def test_two_recorded_episodes_share_clock_and_hold_completed_home_pose():
    from nbv.visualization.playback import RecordedTimeline,SharedClock
    tracks=[RecordedTimeline(_rows(2.,2.)),RecordedTimeline(_rows(4.,2.))]
    clock=SharedClock(tracks);clock.seek(4.)
    a,b=clock.states()
    assert a['simulation_time_s']==b['simulation_time_s']==4.
    assert a['phase']=='home' and b['phase']=='intra_pass'
    assert a['coverage']==.96 and b['coverage']==0.
    clock.seek(6.)
    a,b=clock.states()
    assert a['simulation_time_s']==b['simulation_time_s']==6.
    assert np.array_equal(a['q'],np.zeros(6)) and a['elapsed_s']==5.


def test_slide_export_preserves_exact_hd_dimensions_and_source_metadata(tmp_path):
    from nbv.visualization.slides import slide_figure,save_slide
    from PIL import Image
    fig,ax=slide_figure();ax.plot([0,1],[0,1])
    target=tmp_path/'figure';metadata={'source_run_ids':['synthetic'],'split':'synthetic'}
    save_slide(fig,target,metadata)
    with Image.open(target.with_suffix('.png')) as im:assert im.size==(1920,1080)
    assert target.with_suffix('.pdf').stat().st_size>0
    assert json.loads(target.with_suffix('.json').read_text(encoding='utf-8'))['source_run_ids']==['synthetic']


def test_overlay_rejects_wrong_joint_path_or_one_ulp_time_boundary():
    from nbv.visualization.playback import AuditIndex,RecordedTimeline
    from nbv.collision_audit import segment_id
    track=RecordedTimeline(_rows(2.,2.));s=track.segments[0]
    key=segment_id(271875,s['start'],s['end'])
    interval={'body':'sensor_head','obstacle':'part','start_fraction':.2,'end_fraction':.6,'penetration_mm':1.}
    event={'id':'run:271875_v0','source_run':'run','variant':'271875_v0','part':271875,
           'phase':'transfer','step':0,'geometry_id':key,'time_s':[0.,1.]}
    def make(e):return AuditIndex({'complete':True,'events':[e],'geometries':{e['geometry_id']:{'intervals':[interval]}}})
    assert make(event).validate_timeline('run','271875_v0',track)==1
    with pytest.raises(ValueError,match='geometry'):make(event|{'geometry_id':'wrong'}).validate_timeline('run','271875_v0',track)
    with pytest.raises(ValueError,match='time'):make(event|{'time_s':[0.,float(np.nextafter(1.,2.))]}).validate_timeline('run','271875_v0',track)
