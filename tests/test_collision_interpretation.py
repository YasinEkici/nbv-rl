"""The audit explains the motion lower bound without asserting unmeasured ratios."""
import numpy as np


def _report():
    from nbv.collision_audit import summarize_events
    episodes=[]
    for split in ('val','train'):
        for method in ('ilp_tsp','joint_opt','mcts','sarsa/value_estimate'):
            hits=0 if method in ('ilp_tsp','joint_opt') else (
                1 if (split,method) in (('val','mcts'),('train','sarsa/value_estimate')) else 2)
            episodes.append(dict(id=f'{split}/{method}',part=1,split=split,method=method,
                weight_kind='tail' if method.startswith('sarsa') else 'baseline',
                transfer_count=3,colliding_transfers=hits,any_collision=bool(hits),
                penetrations_mm=[1.]*hits,phase_counts={'transfer':[3,hits],'home':[1,0]}))
    return dict(episodes=episodes,by_part_method=summarize_events(episodes),events=[],geometries={},
                scope='all',sampling_max_joint_degrees=1.,near_mm=10.)


def test_generated_interpretation_uses_rates_including_home_and_records_reversal():
    from nbv.collision_audit import motion_model_interpretation
    text=motion_model_interpretation(_report())
    assert 'robot.movej_time' in text and 'robot.transfer' in text
    assert 'rest-to-rest' in text and 'same unwrapped endpoints' in text
    assert '| val | mcts | 25.00% |' in text
    assert '| val | sarsa/value_estimate | 50.00% |' in text
    assert '| train | mcts | 50.00% |' in text
    assert '| train | sarsa/value_estimate | 25.00% |' in text
    assert 'reverses' in text


def test_interpretation_qualifies_denominator_and_closed_test_extrapolation(tmp_path,monkeypatch):
    import nbv.collision_audit as audit
    text=audit.motion_model_interpretation(_report())
    for phrase in ('denominator','does not prove every hypothesis robust',
                   'analogy from VAL/train','Additional collision-free time was not measured'):
        assert phrase in text
    monkeypatch.setattr(audit,'report_path',lambda scope:tmp_path/'audit.json')
    audit.write_report(_report())
    assert text in (tmp_path/'audit.md').read_text(encoding='utf-8')


def test_existing_movej_time_supports_fixed_endpoint_rest_to_rest_lower_bound():
    from nbv import parts,robot
    cfg=parts.load_yaml('configs/robot_v2.yaml')['motion'];v=cfg['v_rad_s'];a=cfg['a_rad_s2']
    displacements=np.zeros((500,6));displacements[:,0]=np.linspace(0.,10.,500)
    times=robot.movej_time(displacements,v,a)
    assert np.all(np.diff(times)>=0.)
    start=np.zeros(6);end=np.array([1.,0.,0.,0.,0.,0.]);via=np.array([.5,2.,0.,0.,0.,0.])
    direct=robot.movej_time(end-start,v,a)
    detour=robot.movej_time(via-start,v,a)+robot.movej_time(end-via,v,a)
    assert detour>direct
