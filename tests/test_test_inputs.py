"""Input preparation never constructs a held-out environment or evaluates a policy."""
import json
from pathlib import Path
import pytest


def test_dry_run_only_stats_files_and_never_reads_caches(tmp_path, monkeypatch):
    from nbv import test_inputs, env
    source = tmp_path/'cache.npz'
    source.write_bytes(b'not even a valid npz')
    absent = tmp_path/'missing.npz'
    monkeypatch.setattr(test_inputs, 'required_inputs', lambda **kw: [
        {'path': str(source), 'role': 'visibility', 'part': '1'},
        {'path': str(absent), 'role': 'nominal_points', 'part': '1'}])
    monkeypatch.setattr(env, 'NBVEnv', lambda *a, **kw: pytest.fail('constructed environment'))
    monkeypatch.setattr(Path, 'read_bytes', lambda *a: pytest.fail('read cache content'))
    monkeypatch.setattr(Path, 'read_text', lambda *a, **kw: pytest.fail('read file content'))
    report = test_inputs.dry_run()
    assert report['missing'] == [str(absent)]
    assert report['inputs'][0]['size_bytes'] == len(b'not even a valid npz')


def test_inventory_covers_nominal_variant_dense_plans_and_all_checkpoint_files():
    from nbv.test_inputs import required_inputs
    entries = required_inputs(variants=['1_v0', '1_v1'], run_ids=['run-1'])
    paths = {e['path'] for e in entries}
    assert 'data/cache/visibility/nominal/points_1.npz' in paths
    assert 'data/cache/visibility/1_v1_n128.npz' in paths
    assert 'data/cache/robot/nominal/1_n48.npz' in paths
    assert 'data/cache/plans/test_1_n48.json' in paths
    assert 'outputs/run-1/provenance.json' in paths
    assert 'results/d_plan_v2.csv' in paths
    assert 'results/selected_settings_v1.json' in paths


def test_input_manifest_rejects_missing_and_changed_bytes(tmp_path):
    from nbv.test_inputs import make_input_manifest, verify_input_manifest
    source = tmp_path/'input'; source.write_bytes(b'original')
    inventory = {'inputs': [{'path':str(source),'role':'synthetic','exists':True,'size_bytes':8}], 'missing':[]}
    manifest = make_input_manifest(inventory)
    verify_input_manifest(manifest, inventory)
    source.write_bytes(b'altered!')
    with pytest.raises(ValueError, match='SHA-256'):
        verify_input_manifest(manifest, inventory)
    with pytest.raises(ValueError, match='missing'):
        make_input_manifest({'inputs':[], 'missing':['missing']})


def test_nominal_plan_preparation_uses_frozen_settings_without_environment(monkeypatch):
    from nbv import test_inputs, env, baselines
    nominal = object()
    monkeypatch.setattr(env, 'NBVEnv', lambda *a, **kw: pytest.fail('constructed environment'))
    monkeypatch.setattr(test_inputs, 'load_nominal', lambda fid: nominal)
    monkeypatch.setattr(baselines, 'ilp_tsp_plan', lambda n: ([1,2], {'ilp_status':'OPTIMAL','ilp_gap':0.}))
    def solve(n, **kw):
        assert n is nominal and kw == dict(hint=[1,2], time_limit_s=300, workers=8)
        return [2,1], {'joint_status':'OPTIMAL','joint_bound_s':4.,'joint_obj_s':4.}
    monkeypatch.setattr(baselines, 'joint_opt_plan', solve)
    result = test_inputs.solve_nominal(1)
    assert result['ilp_plan'] == [1,2] and result['joint_info']['joint_bound_s'] == 4.
