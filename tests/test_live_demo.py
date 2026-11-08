"""VAL-only live demo: guards, exact source matching and real loopback smoke."""
import copy
import socket
import urllib.request
from pathlib import Path
import numpy as np
import pytest


@pytest.mark.parametrize('part,variant',[(178195,0),(486859,0),(79729,4),(46262,0),(271875,5)])
def test_live_rejects_non_val_and_invalid_variants_before_environment(part,variant,monkeypatch):
    from nbv.visualization.live import allowed_variant
    import nbv.env
    monkeypatch.setattr(nbv.env,'NBVEnv',lambda *a,**kw:pytest.fail('environment constructed'))
    with pytest.raises(ValueError,match='VAL'):allowed_variant(part,variant)


def test_corrupted_live_reference_is_explicit_mismatch_not_relaxed():
    from nbv.visualization.live import verify_live
    original={'sequence':[1,2],'cycle_time':12.,'success':True,'normalized_cycle_time':1.2}
    assert verify_live(original,original)['exact']
    wrong=original|{'cycle_time':float(np.nextafter(12.,np.inf))}
    verdict=verify_live(original,wrong)
    assert not verdict['exact'] and 'cycle_time' in verdict['mismatches']


@pytest.mark.parametrize('method,variant',[
    ('greedy_cost',0),('mcts',1),('sarsa/linear',0),('sarsa/value_estimate',0),('regq/linear',0),
    ('ilp_tsp',0),('joint_opt',0),
])
def test_real_val_live_methods_match_saved_episodes_headless(method,variant,monkeypatch):
    from nbv.visualization.live import LiveSession,SourceStore
    from nbv.visualization.replay import digest
    registry=digest('results/registry.csv');manifest=digest('results/demo/manifest.json')
    original=Path.open
    def no_experiment_writes(path,mode='r',*a,**kw):
        if any(flag in mode for flag in ('w','a','x','+')) and ('outputs' in path.parts or path.name=='registry.csv'):
            pytest.fail('live demo attempted an experiment write')
        return original(path,mode,*a,**kw)
    monkeypatch.setattr(Path,'open',no_experiment_writes)
    store=SourceStore()
    required=[Path(f'data/cache/visibility/271875_v{variant}_n48.npz'),
        Path('outputs')/store.row(method,0)['run_id']/'episodes.jsonl',
        Path('outputs')/store.row('ilp_tsp',0)['run_id']/'episodes.jsonl']
    if '/' in method:required.append(Path('outputs')/store.row(method,0)['run_id']/'weights.npz')
    if not all(p.exists() for p in required):pytest.skip('original VAL caches/checkpoints are not distributed with source')
    session=LiveSession(store,271875,variant,method,0)
    while not session.done:session.step()
    assert session.verification['exact'],session.verification
    assert session.summary['sequence']==session.expected['sequence']
    assert session.verification['matched_fields']
    assert registry==digest('results/registry.csv') and manifest==digest('results/demo/manifest.json')


def test_static_player_loads_hash_verified_bundle_and_refuses_tampered_manifest(tmp_path,monkeypatch):
    from nbv.visualization.live import recorded_session
    from nbv.visualization.demo import write_manifest
    import nbv.visualization.live as live
    monkeypatch.setattr(live,'NBVEnv',lambda *a,**kw:pytest.fail('static player constructed environment'))
    monkeypatch.setattr(np,'load',lambda *a,**kw:pytest.fail('static player opened numerical cache/checkpoint'))
    data=recorded_session(271875,'sarsa/linear')
    assert data['mode']=='recorded bundle' and data['steps'][-1]['kind']=='home'
    (tmp_path/'common.json').write_text('{}',encoding='utf-8');write_manifest(tmp_path,commit='synthetic')
    (tmp_path/'common.json').write_text('tampered',encoding='utf-8')
    with pytest.raises(ValueError,match='hash'):recorded_session(271875,'sarsa/linear',directory=tmp_path)


def test_live_server_loopback_http_200_and_shutdown():
    pytest.importorskip('viser')
    from nbv.visualization.live import create_server
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    server=create_server(port=port,load_initial=False)
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{server.get_port()}',timeout=10) as response:
            assert response.status==200
    finally:server.stop()


def test_point_colors_distinguish_all_four_states_and_share_live_recorded_meaning():
    from nbv.visualization.live_scene import point_colors,POINT_COLORS
    colors=point_colors(np.array([1,1,0,0],bool),np.array([0,1,0,0],bool),np.array([1,1,1,0],bool))
    assert np.array_equal(colors,np.array([POINT_COLORS[k] for k in ('measured','new','available','unmeasurable')]))
    assert len({tuple(c) for c in colors})==4


def test_live_checkpoint_hash_tampering_is_rejected_before_inference():
    from nbv.visualization.live import SourceStore
    store=SourceStore();rid=store.row('sarsa/linear',0)['run_id']
    if not (Path('outputs')/rid/'weights.npz').exists():pytest.skip('original checkpoint unavailable')
    store.pretest=copy.deepcopy(store.pretest)
    store.pretest['runs'][rid]['sha256']['weights.npz']='0'*64
    with pytest.raises(ValueError,match='SHA-256 mismatch'):store.weights('sarsa/linear',0)


def test_policy_stop_still_produces_home_row_and_allows_terminal_verification():
    from types import SimpleNamespace
    from nbv.visualization.live_scene import DemoController
    controller=DemoController.__new__(DemoController)
    controller.recorded=False;controller.home_done=False
    session=SimpleNamespace(done=False,steps=[{'t_home':2.,'coverage':.8}],
        joints=[{'q_end':[1.]*6}],summary={'cycle_time':10.})
    def finish_without_action():session.done=True;return None
    session.step=finish_without_action;controller.session=session
    controller.scene=SimpleNamespace(home=np.zeros(6))
    row=controller._next_row()
    assert row['kind']=='home' and row['cumulative_time_s']==10.
    assert controller._next_row() is None
