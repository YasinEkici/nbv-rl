"""Exercise real initialized VAL sessions, not merely the HTTP listener."""
import socket
from pathlib import Path
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


def _require_live_material():
    """Live integration needs originals; recorded playback remains source-portable."""
    if not Path('data/cache/visibility/271875_v0_n48.npz').is_file():
        pytest.skip('live initialization requires original VAL visibility/cache material')
    from nbv.visualization.live import SourceStore,METHODS
    store=SourceStore()
    required=[Path('outputs')/store.row(method,0)['run_id']/'episodes.jsonl' for method in METHODS]
    if not all(path.is_file() for path in required):
        pytest.skip('live initialization requires original canonical VAL episode records')


@pytest.mark.parametrize('method',[
    'greedy_cost','mcts','sarsa/linear','sarsa/value_estimate','regq/linear','ilp_tsp','joint_opt'])
def test_live_first_session_has_scene_chart_and_no_swallowed_error(method):
    _require_live_material()
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


@pytest.mark.parametrize('recorded',[False,True])
def test_compact_panel_uses_readable_choices_and_collapsed_details(recorded):
    if not recorded:_require_live_material()
    server=_server(recorded=recorded)
    try:
        c=server._nbv_controller
        with c.lock:
            c.reset()
            assert c.summary_panel.content
            assert not c.details_folder.expand_by_default
            assert c.error_text==''
            options=c.method.handle.options
            assert not any('/linear' in name or 'greedy_cost' in name for name in options)
            assert 'Kapsama' in c.summary_panel.content
    finally:server.stop()


def test_compact_recorded_panel_keeps_active_collision_warning_visible():
    server=_server(recorded=True)
    try:
        c=server._nbv_controller
        with c.lock:
            c.reset();episode=c.episodes[0]
            hit=c.audit.events[(episode['source_run_id'],episode['variant'])][0]
            c.seek((hit['start_s']+hit['end_s'])/2)
            assert 'çarpışma (' in c.status.content
    finally:server.stop()
