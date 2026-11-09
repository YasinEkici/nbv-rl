"""Exercise real initialized VAL sessions, not merely the HTTP listener."""
import socket
import pytest


def _server(recorded=False):
    from nbv.visualization.live import create_server
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    return create_server(port,recorded=recorded,load_initial=False)


def _scene_ready(server,scene):
    nodes=server.scene._handle_from_node_name
    assert scene.api.prefix+'/part' in nodes
    assert scene.api.prefix+'/samples' in nodes
    assert scene.cloud is not None and scene.capsules
    assert len(scene.faces)>0 and len(scene.points)==20000


@pytest.mark.parametrize('method',[
    'greedy_cost','mcts','sarsa/linear','sarsa/value_estimate','regq/linear','ilp_tsp','joint_opt'])
def test_live_first_session_has_scene_chart_and_no_swallowed_error(method):
    server=_server()
    try:
        c=server._nbv_controller
        with c.lock:
            c.method.value=method;c.reset()
            assert c.session is not None,c.status.content
            assert c.session.method==method
            _scene_ready(server,c.scene)
            assert len(c.chart.episodes)==7 and len(c.chart.handle.figure.data)==7
            assert 'HATA' not in c.status.content and not getattr(c,'error_text','')
    finally:server.stop()


@pytest.mark.parametrize('comparison',[False,True])
def test_recorded_first_session_and_distinct_comparison_have_ready_chart(comparison):
    server=_server(recorded=True)
    try:
        c=server._nbv_controller
        with c.lock:
            c.method.value='greedy_cost';c.other.value='sarsa/value_estimate'
            c.comparison.value=comparison;c.reset()
            assert c.clock is not None,c.status.content
            assert len(c.episodes)==(2 if comparison else 1)
            for scene in c.scenes:_scene_ready(server,scene)
            assert len(c.chart.episodes)==7 and len(c.chart.handle.figure.data)==7
            assert 'HATA' not in c.status.content and not getattr(c,'error_text','')
            if comparison:
                assert c.episodes[0]['method']!=c.episodes[1]['method']
                assert c.episodes[0]['source_run_id']!=c.episodes[1]['source_run_id']
    finally:server.stop()
