"""Synthetic registry only: importing and rejecting never constructs a real environment."""
import json
import pytest


def synthetic_rows():
    return [dict(plan='002', status='canonical', part_split='val', method=m, seed=str(s),
                 env_version='v4', split_version='v1', budget_steps='2000000', run_id=f'{m}-{p}-{s}',
                 extra_json=json.dumps({'tag': 'p002', 'feature_count': 11, 'phi': p, 'tail_count': 200000}))
            for m, p in [('sarsa', 'linear'), ('regq', 'linear'), ('myopic', 'linear'),
                         ('sarsa', 'none'), ('sarsa', 'convex'), ('sarsa', 'value_estimate')] for s in range(10)]


@pytest.mark.parametrize('condition', ['flag', 'dirty', 'incomplete', 'duplicate', 'test_row', 'missing_weights', 'budget'])
def test_test_opening_rejects_unsafe_synthetic_inputs(condition):
    from nbv.test_evaluation import validate_opening
    rows, state, flag, exists = synthetic_rows(), {'dirty': False}, True, lambda r: True
    if condition == 'flag': flag = False
    if condition == 'dirty': state['dirty'] = True
    if condition == 'incomplete': rows.pop()
    if condition == 'duplicate': rows.append(rows[0])
    if condition == 'test_row': rows.append({'part_split': 'test'})
    if condition == 'missing_weights': exists = lambda r: False
    if condition == 'budget': rows[0]['budget_steps'] = '200000'
    with pytest.raises(ValueError):
        validate_opening(flag, state, rows, exists)


def test_test_opening_accepts_exact_complete_synthetic_official_set():
    from nbv.test_evaluation import validate_opening
    rows = synthetic_rows()
    assert len(validate_opening(True, {'dirty': False}, rows, lambda r: True)) == 60


def test_test_opening_claim_refuses_reentry_even_without_registry_results(tmp_path):
    from nbv.test_evaluation import claim_opening
    path = tmp_path / 'claim.json'
    claim_opening(path, {'commit': 'abc'})
    with pytest.raises(FileExistsError):
        claim_opening(path, {'commit': 'abc'})


def test_refused_opening_never_loads_an_environment(monkeypatch):
    from nbv import test_evaluation
    monkeypatch.setattr(test_evaluation.evaluation, 'git_state', lambda: {'dirty': False})
    monkeypatch.setattr(test_evaluation, 'load_rows', lambda _: synthetic_rows())
    monkeypatch.setattr(test_evaluation, 'execute', lambda *a, **kw: pytest.fail('evaluation opened'))
    with pytest.raises(ValueError):
        test_evaluation.open_test(understand=False)


def test_future_joint_reference_keeps_ilp_hint_and_solver_bound(monkeypatch):
    from nbv import baselines
    from nbv.test_evaluation import prepare_joint
    nominal = object()
    monkeypatch.setattr(baselines, 'ilp_tsp_plan', lambda n: ([3, 4], {'ilp_status': 'OPTIMAL'}))
    def solve(n, **kwargs):
        assert n is nominal and kwargs['hint'] == [3, 4]
        assert kwargs['time_limit_s'] == 300 and kwargs['workers'] == 8
        return [4, 3], {'joint_bound_s': 4., 'joint_obj_s': 5.}
    monkeypatch.setattr(baselines, 'joint_opt_plan', solve)
    plan, info = prepare_joint(nominal)
    assert plan == [4, 3] and info['joint_bound_s'] == 4.


@pytest.mark.parametrize('damage', [None, 'missing_history', 'partial', 'dirty', 'shape', 'nonfinite', 'mismatched_weights', 'budget'])
def test_checkpoint_guard_checks_actual_completion_without_real_caches(tmp_path, damage):
    import numpy as np
    from nbv.test_evaluation import checkpoint_complete
    training = {'history': [{'step': s, 'weight_norm': 0., 'normalized_cycle_time': 1.}
                             for s in range(100000, 2000001, 100000)],
                'policy_weights': [0.] * 11, 'final_weights': [0.] * 11}
    provenance = {'dirty': damage == 'dirty', 'env_version': 'v4', 'split_version': 'v1'}
    config = {'budget': 200000 if damage == 'budget' else 2000000, 'features': 11}
    if damage == 'partial': training['history'].pop()
    tail = np.zeros(16 if damage == 'shape' else 11)
    if damage == 'nonfinite': tail[0] = np.nan
    if damage == 'mismatched_weights': tail[0] = 1.
    np.savez(tmp_path / 'weights.npz', tail=tail, final=np.zeros(11))
    (tmp_path / 'provenance.json').write_text(json.dumps(provenance))
    (tmp_path / 'config.json').write_text(json.dumps(config))
    if damage != 'missing_history':
        (tmp_path / 'training.json').write_text(json.dumps(training))
    assert checkpoint_complete(tmp_path) == (damage is None)
