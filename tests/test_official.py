"""Official tasks are fixed before running and stop gates preserve completed rows."""
import pytest


def test_official_tasks_have_sixty_canonical_and_ten_exploratory_seeds():
    from nbv.official import official_tasks, check_completed
    from nbv.experiment import StopRun
    tasks = official_tasks({'sarsa': {'setting': {'alpha': .03, 'lam': .5, 'epsilon': 'constant'}},
                            'regq': {'setting': {'alpha': .01, 'eta': .001, 'epsilon': 'constant'}},
                            'myopic': {'setting': {'alpha': .03, 'phi_alpha': 50., 'epsilon': 'constant'}}})
    assert len(tasks) == 70
    assert sum(t['_status'] == 'canonical' for t in tasks) == 60
    assert all(t['budget'] == 2000000 for t in tasks)
    assert all(t['features'] == 11 or t['_status'] == 'exploratory' for t in tasks)
    assert {t['seed'] for t in tasks} == set(range(10))
    results = [{'task': tasks[0] | {'seed': i}, 'diverged': True} for i in range(6)]
    with pytest.raises(StopRun, match='diverg'):
        check_completed(results)


def test_official_deadline_stops_before_launching_work():
    from nbv.experiment import run_tasks, StopRun
    with pytest.raises(StopRun, match='deadline'):
        run_tasks([], status='canonical', tag='p002', max_wall_s=0)
