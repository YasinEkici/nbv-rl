import dataclasses
import inspect

import numpy as np
import pytest

from nbv import env as nbv_env
from nbv import features, robot

P, N, K = 8, 60, 3


def synthetic(variant: str = "1_v0", seed: int = 0, real_seed: int | None = None) -> nbv_env.EpisodeData:
    """Small part: 8 passes (2 unreachable), 60 samples; real visibility = nominal with a few flips."""
    rng = np.random.default_rng(seed)
    home = np.radians([0.0, -90.0, 90.0, -90.0, -90.0, 0.0])
    start = home + rng.uniform(-2.5, 2.5, size=(P, 6))
    q = start[:, None, :] + np.linspace(0, 0.2, K)[None, :, None]
    reach = np.ones(P, bool)
    reach[[2, 5]] = False
    q[~reach] = np.nan
    nom_vis = rng.random((P, N)) < 0.3
    rr = np.random.default_rng(seed + 1 if real_seed is None else real_seed)
    real_vis = nom_vis ^ (rr.random((P, N)) < 0.05)
    v, a = 1.05, 1.4
    ptp_pi = robot.movej_time(robot.wrap_pi(q[:, -1][:, None] - q[:, 0][None]), v, a)
    nominal = nbv_env.NominalModel(
        points=rng.normal(size=(N, 3)), normals=rng.normal(size=(N, 3)), visibility=nom_vis, reachable=reach,
        union=nom_vis[reach].any(0), q_plan=q, direction=rng.normal(size=(P, 3)), scan_time=rng.uniform(2, 3, P),
        ptp_pi=ptp_pi, from_home_pi=robot.movej_time(robot.wrap_pi(q[:, 0] - home), v, a),
        to_home_pi=robot.movej_time(robot.wrap_pi(home - q[:, -1]), v, a),
    )
    real = nbv_env.RealPart(real_vis, real_vis[reach].any(0), real_vis[reach].any(0))
    return nbv_env.EpisodeData(variant, 48, nominal, real)


def make_env(data=None, **kw):
    data = data or synthetic()
    return nbv_env.NBVEnv([data.variant_id], loader=lambda vid: data, **kw)


def greedy_episode(e, seed=None):
    obs, info = e.reset(seed=seed, options={"variant": e.variants[0]})
    infos, rewards, done = [], [], False
    while not done:
        gain = features.nominal_gain(obs, e.nominal).astype(float)
        gain[~e.action_masks()] = -1
        obs, r, term, trunc, info = e.step(int(np.argmax(gain)))
        infos.append(info["agent"] | info["eval"])
        rewards.append(r)
        done = term or trunc
    return infos, rewards


def test_coverage_is_monotonic_and_bounded():
    infos, _ = greedy_episode(make_env())
    c = [i["coverage"] for i in infos]
    assert all(b >= a for a, b in zip(c, c[1:]))
    assert 0.0 <= c[0] and c[-1] <= 1.0


def test_return_is_minus_cycle_time_plus_bonus():
    for data in (synthetic(), synthetic(seed=3)):
        e = make_env(data)
        infos, rewards = greedy_episode(e)
        assert np.isclose(sum(i["delta_coverage"] for i in infos), infos[-1]["coverage"])
        assert all(set(i["reward_parts"]) == {"move", "scan", "home", "bonus"} for i in infos)
        expected = -e.cycle_time + (e.bonus if infos[-1]["success"] else 0.0)
        assert np.isclose(sum(rewards), expected)
    # Random policies, including episodes that end at K_max without success.
    for seed in range(5):
        e = make_env()
        e.k_max = 3
        obs, _ = e.reset(seed=seed, options={"variant": e.variants[0]})
        rng, total, done = np.random.default_rng(seed), 0.0, False
        while not done:
            obs, r, term, trunc, info = e.step(int(rng.choice(np.flatnonzero(e.action_masks()))))
            total += r
            done = term or trunc
        assert np.isclose(total, -e.cycle_time + e.bonus * info["agent"]["success"])


def test_info_separates_agent_and_eval_fields():
    e = make_env()
    _, info = e.reset(options={"variant": e.variants[0]})
    assert set(info) == {"agent", "eval"}
    assert set(info["eval"]) == {"coverage", "coverage_nsc", "coverage_raw"}
    _, _, _, _, info = e.step(0)
    assert set(info["eval"]) == {"coverage", "coverage_nsc", "coverage_raw", "delta_coverage", "coverage_gap_nominal_minus_real"}
    assert not set(info["agent"]) & set(info["eval"])
    assert "coverage" not in info["agent"]


def test_masked_action_raises():
    e = make_env()
    e.reset(options={"variant": e.variants[0]})
    with pytest.raises(ValueError):
        e.step(2)  # unreachable
    e.step(0)
    with pytest.raises(ValueError):
        e.step(0)  # already used


def test_same_seed_same_episode():
    a, _ = greedy_episode(make_env(), seed=3)
    b, _ = greedy_episode(make_env(), seed=3)
    assert [i["pass"] for i in a] == [i["pass"] for i in b]
    assert [i["cycle_time"] for i in a] == [i["cycle_time"] for i in b]


def test_step_times_sum_to_cycle_time():
    e = make_env()
    infos, _ = greedy_episode(e)
    total = sum(i["t_move"] + i["t_scan"] + i["t_home"] for i in infos)
    assert np.isclose(total, e.cycle_time)
    assert np.isclose(infos[-1]["cycle_time"], e.cycle_time)


def test_transfers_respect_wrapped_lower_bound():
    data = synthetic()
    e = make_env(data)
    infos, _ = greedy_episode(e)
    nom, prev = data.nominal, None
    for i in infos:
        lb = nom.from_home_pi[i["pass"]] if prev is None else nom.ptp_pi[prev, i["pass"]]
        assert i["t_move"] >= lb - 1e-9
        prev = i["pass"]
    assert infos[-1]["t_home"] >= nom.to_home_pi[prev] - 1e-9


def test_k_max_terminates_the_episode():
    data = synthetic()
    # Real visibility that never reaches the target: each pass sees a single distinct sample.
    real_vis = np.zeros((P, N), bool)
    real_vis[np.arange(P), np.arange(P)] = True
    real_vis[0, 10:30] = True
    data = dataclasses.replace(data, real=nbv_env.RealPart(real_vis, real_vis[data.nominal.reachable].any(0), real_vis.any(0)))
    e = make_env(data)
    e.k_max = 3
    e.reset(options={"variant": data.variant_id})
    for k, a in enumerate((0, 1, 3)):
        _, _, term, trunc, info = e.step(a)
    # The budget is part of the task (D-038): a real end, not a time limit.
    assert term and not trunc and e.state()["step"] == 3 and info["agent"]["t_home"] > 0
    assert not info["agent"]["success"] and info["agent"]["reward_parts"]["bonus"] == 0.0


def test_hand_computed_joint_tracking():
    data = synthetic()
    e = make_env(data)
    e.reset(options={"variant": data.variant_id})
    q = e.q_home.copy()
    expected = 0.0
    for a in (1, 4):
        sweep = data.nominal.q_plan[a]
        target = sweep[0].copy()
        for j in range(6):  # closest 2*pi-equivalent of the start (all within limits here)
            target[j] = q[j] + robot.wrap_pi(sweep[0, j] - q[j])
        t = robot.movej_time(target - q, e.v, e.a)
        _, _, _, _, info = e.step(a)
        assert np.isclose(info["agent"]["t_move"], t)
        expected += t + data.nominal.scan_time[a]
        q = target + (sweep[-1] - sweep[0])
        assert np.allclose(e.state()["q"], q)
    _, info = e.finish()
    expected += robot.movej_time(e.q_home - q, e.v, e.a)
    assert np.isclose(e.cycle_time, expected)


def test_transfer_keeps_sweep_inside_limits():
    lim = np.radians(360.0)
    qlim = np.array([[-lim] * 6, [lim] * 6])
    q = np.zeros(6)
    q[0] = np.radians(350.0)
    sweep = np.zeros((3, 6))
    sweep[:, 0] = np.radians([-20.0, 0.0, 30.0])  # +360 would be closest but ends at 390 deg
    _, out = robot.transfer(q, sweep, qlim, 1.05, 1.4)
    assert out[:, 0].max() <= lim and out[:, 0].min() >= -lim
    assert np.allclose(np.diff(out[:, 0]), np.diff(sweep[:, 0]))


def test_evaluate_sequence_matches_stepping():
    e = make_env()
    infos, rewards = greedy_episode(e)
    seq = [i["pass"] for i in infos]
    res = nbv_env.evaluate_sequence(make_env(), e.variants[0], seq + [7])
    assert res["passes_executed"] == len(seq)
    assert [s["coverage"] for s in res["steps"]] == [i["coverage"] for i in infos]
    assert np.isclose(res["cycle_time"], e.cycle_time)
    assert np.isclose(res["return"], sum(rewards))


def test_evaluate_sequence_censored_when_sequence_ends():
    res = nbv_env.evaluate_sequence(make_env(), "1_v0", [0])
    assert res["censored"] or res["success"]
    if res["censored"]:
        assert res["steps"][-1]["t_home"] > 0 and res["passes_to_target"] is None


def test_features_do_not_depend_on_privileged_data():
    a, b = synthetic(real_seed=11), synthetic(real_seed=99)
    assert not np.array_equal(a.real.visibility, b.real.visibility)
    assert a.nominal.visibility is not b.nominal.visibility and np.array_equal(a.nominal.visibility, b.nominal.visibility)
    obs = {"covered": np.zeros(N, np.int8), "used": np.zeros(P, np.int8), "q": np.radians([0, -90, 90, -90, -90, 0.0])}
    for f in (features.nominal_gain, lambda o, n: features.move_time_estimate(o, n, 1.05, 1.4)):
        assert np.array_equal(f(obs, a.nominal), f(obs, b.nominal), equal_nan=True)
    for name, fn in inspect.getmembers(features, inspect.isfunction):
        if fn.__module__ == "nbv.features":
            params = list(inspect.signature(fn).parameters)
            assert "env" not in params and "real" not in params, name


def test_observation_has_no_privileged_fields():
    e = make_env()
    obs, _ = e.reset(options={"variant": e.variants[0]})
    assert set(obs) == {"covered", "used", "q", "step", "nominal_coverage"}
    assert not any(isinstance(getattr(e.nominal, f.name), nbv_env.RealPart) for f in dataclasses.fields(e.nominal))
    assert e.observation_space.contains(obs)


def test_unsupported_pose_raises():
    e = make_env()
    with pytest.raises(NotImplementedError):
        e.reset(options={"variant": e.variants[0], "pose_id": 3})


def test_plan_repair_adds_passes_until_done():
    from nbv import baselines

    e = make_env()
    repair = baselines.repair_policy(e.v, e.a)
    res = nbv_env.evaluate_sequence(e, e.variants[0], [0], repair=repair)
    assert res["repair_passes"] == sum(s["repair"] for s in res["steps"]) == res["passes_executed"] - 1
    assert not res["steps"][0]["repair"] and all(s["repair"] for s in res["steps"][1:])
    total = sum(s["t_move"] + s["t_scan"] + s["t_home"] for s in res["steps"])
    assert np.isclose(total, res["cycle_time"])
    # Without repair the same plan is censored at its end (unless the single pass already succeeds).
    plain = nbv_env.evaluate_sequence(make_env(), e.variants[0], [0])
    assert plain["repair_passes"] == 0 and plain["passes_executed"] == 1


def test_repair_of_empty_plan_is_closed_loop_greedy_cost():
    from nbv import baselines

    e = make_env()
    repair = baselines.repair_policy(e.v, e.a)
    res = nbv_env.evaluate_sequence(e, e.variants[0], [], repair=repair)
    obs, _ = e.reset(options={"variant": e.variants[0]})
    seq, done = [], False
    while not done:
        a = baselines.greedy_cost_action(obs, e.nominal, e.action_masks(), e.v, e.a)
        if a is None:
            break
        obs, _, term, trunc, _ = e.step(a)
        seq.append(a)
        done = term or trunc
    assert [s["pass"] for s in res["steps"]] == seq


def test_policy_returning_none_ends_the_episode_censored():
    """D-014 / PROTOCOL: a policy (or repair) that returns no action ends the episode early, censored."""
    from nbv import evaluation

    e = make_env()
    calls = []

    def one_pass_then_stop(obs, nominal, mask):
        calls.append(int(obs["step"][0]))
        return int(np.flatnonzero(mask)[0]) if not calls[:-1] else None

    res = nbv_env.evaluate_policy(e, e.variants[0], lambda env: one_pass_then_stop)
    assert res["passes_executed"] == 1 < e.k_max and calls == [0, 1]
    assert res["censored"] and not res["success"] and res["passes_to_target"] is None
    st = res["steps"][0]
    assert st["t_home"] > 0  # the arm returns home when the episode is cut
    assert np.isclose(res["cycle_time"], st["t_move"] + st["t_scan"] + st["t_home"])
    m = evaluation.episode_metrics(res)
    assert not m["success"] and m["passes_to_95"] is None and np.isclose(m["cycle_time"], res["cycle_time"])
