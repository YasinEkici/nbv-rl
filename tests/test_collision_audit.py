"""Collision diagnosis uses dense executed paths, exact geometry and a hard waypoint gate."""
import numpy as np
import pytest


def test_joint_sampling_preserves_multiturn_endpoints_and_one_degree_bound():
    from nbv.collision_audit import sample_joint_segment
    a=np.array([0.,0.,0.,0.,0.,0.]);b=np.array([2*np.pi,0.,0.,0.,0.,0.])
    q,f=sample_joint_segment(a,b,max_step_deg=1.)
    assert np.array_equal(q[0],a) and np.array_equal(q[-1],b)
    assert len(q)==361 and f[0]==0 and f[-1]==1
    assert np.abs(np.diff(q,axis=0)).max()<=np.deg2rad(1.)+np.finfo(float).eps*8


def test_dense_segment_detects_interior_collision_and_reports_body_obstacle():
    from nbv.collision_audit import inspect_segment
    def clearance(q):return {'sensor_head':(100.,abs(float(q[0])-.5)*20-2.)}
    a=np.zeros(6);b=np.zeros(6);b[0]=1.
    result=inspect_segment(a,b,clearance,max_step_deg=1.,near_mm=10.)
    assert result['collided'] and result['worst']['obstacle']=='table'
    assert result['worst']['body']=='sensor_head' and result['worst']['penetration_mm']>1.8
    assert result['intervals'] and not result['endpoint_collision']


def test_negative_waypoint_clearance_stops_without_hiding_small_penetration():
    from nbv.collision_audit import require_clear_waypoint,WaypointCollision
    with pytest.raises(WaypointCollision,match='sensor_head'):
        require_clear_waypoint({'sensor_head':(-1e-12,100.)},{'part':1,'pass':2,'sample':3})
    require_clear_waypoint({'base':(30.,np.inf),'sensor_head':(5.,5.)},{'part':1})


def test_audit_rejects_test_part_before_geometry_is_opened():
    from nbv.collision_audit import allowed_part
    manifest={'parts':[{'file_id':1,'split':'val'},{'file_id':2,'split':'train'},{'file_id':3,'split':'test'}]}
    assert allowed_part(2,manifest)=='train'
    with pytest.raises(ValueError,match='test'):allowed_part(3,manifest)


def test_executed_segments_keep_return_home_and_four_scan_subsegments():
    from nbv.collision_audit import episode_segments
    q=np.zeros((5,6));q[:,0]=np.arange(5)/10
    record={'steps':[{'pass':2,'t_move':2.,'t_scan':4.,'t_home':3.}],
            'joint_trace':[{'q_start':np.zeros(6).tolist(),'q_samples':q.tolist(),
                            'q_end':q[-1].tolist(),'elapsed_before_home':6.}]}
    segments=list(episode_segments(record,np.zeros(6)))
    assert [s['phase'] for s in segments]==['transfer']+['intra_pass']*4+['home']
    assert [s['time_s'] for s in segments]==[[0.,2.],[2.,3.],[3.,4.],[4.,5.],[5.,6.],[6.,9.]]
    assert np.array_equal(segments[-1]['start'],q[-1])


def test_collision_summary_retains_every_episode_denominator():
    from nbv.collision_audit import summarize_events
    episodes=[{'id':str(i),'part':1,'split':'val','method':'m','weight_kind':'tail',
               'transfer_count':2,'colliding_transfers':i,'any_collision':bool(i),
               'penetrations_mm':[3.] if i else []} for i in range(2)]
    row=summarize_events(episodes)[0]
    assert row['episodes']==2 and row['transfers']==4
    assert row['colliding_transfer_rate']==.25 and row['colliding_episode_rate']==.5


def test_collision_audit_limitations_note_is_generated_without_environment_access():
    from nbv.report_common import collision_limitation
    text=collision_limitation()
    assert 'collision_audit_v1.md' in text and 'sampled' in text and 'D-071' in text


def test_worst_collision_views_are_distinct_paths_and_reject_closed_parts():
    from nbv.visualization.audit_scene import worst_events
    events=[{'geometry_id':str(i//2),'penetration_mm':10-i,'part':271875} for i in range(6)]
    assert [e['geometry_id'] for e in worst_events(events)]==['0','1','2']
    with pytest.raises(ValueError,match='test'):worst_events([events[0]|{'part':178195}])


def test_single_thread_query_preserves_every_pose_clearance_bit():
    from nbv.collision_audit import SingleThreadPartDistance
    from nbv import collision,parts,robot
    import trimesh
    mesh=trimesh.creation.box([140.,140.,140.]);mesh.apply_translation([0.,0.,70.])
    original=collision.PartDistance(mesh);serial=SingleThreadPartDistance(mesh)
    model=robot.make_robot(parts.load_yaml('configs/robot_v2.yaml'))
    geom=collision.ArmGeometry.from_config(parts.load_yaml('configs/collision_v1.yaml'))
    rng=np.random.default_rng(69071)
    for q in rng.uniform(-np.pi,np.pi,(20,6)):
        a=collision.pose_clearance(model,q,original,geom,geom.box_points())
        b=collision.pose_clearance(model,q,serial,geom,geom.box_points())
        assert a.keys()==b.keys()
        assert np.asarray(list(a.values()),dtype='<f8').tobytes()==np.asarray(list(b.values()),dtype='<f8').tobytes()


def test_transition_rate_includes_return_home_as_a_transfer():
    from nbv.collision_audit import summarize_events
    episode={'id':'x','part':1,'split':'val','method':'m','weight_kind':'tail',
        'transfer_count':2,'colliding_transfers':0,'any_collision':True,'penetrations_mm':[5.],
        'phase_counts':{'transfer':[2,0],'home':[1,1],'intra_pass':[8,0]}}
    row=summarize_events([episode])[0]
    assert row['transfers']==3 and row['colliding_transfers']==1
    assert row['colliding_transfer_rate']==1/3
    assert row['home_transfers']==1 and row['colliding_home_transfers']==1


def test_deepest_views_include_all_bodies_and_deduplicate_identical_states():
    from nbv.collision_audit import deepest_events
    a={'body':'forearm','obstacle':'table','penetration_mm':80.,'worst_q_rad':[0.]*6}
    b={'body':'sensor_head','obstacle':'part','penetration_mm':70.,'worst_q_rad':[.1]*6}
    c={'body':'wrist_1','obstacle':'part','penetration_mm':60.,'worst_q_rad':[.2]*6}
    report={'events':[{'part':1,'geometry_id':'a','id':'1'},{'part':1,'geometry_id':'a','id':'2'},
                      {'part':1,'geometry_id':'b','id':'3'}],
            'geometries':{'a':{'intervals':[a,b]},'b':{'intervals':[c]}}}
    result=deepest_events(report)
    assert [e['body'] for e in result]==['forearm','sensor_head','wrist_1']
    assert [e['penetration_mm'] for e in result]==[80.,70.,60.]
