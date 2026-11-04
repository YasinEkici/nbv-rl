"""Correctness of the Phase 1 agents (Plan 002, section 2.1 and Slice 1).

Reference solutions are computed here, without the agent code: the exact online lambda-return
algorithm (true online SARSA(lambda) must reproduce its weights), closed-form values of small
Markov chains, and value iteration.
"""

import numpy as np
import pytest

from nbv import agents
from nbv.agents.linear import EpsilonSchedule, LinearAgent
from nbv.agents.myopic import MyopicAgent
from nbv.agents.qlearning import RegQAgent
from nbv.agents.sarsa import TrueOnlineSarsaAgent
from test_env import make_env

# --- Small problems ----------------------------------------------------------------------------


def run_fixed_episode(agent, xs, rewards):
    """Feed one episode with a single action per state; xs[t] is phi(S_t, A_t) of shape (d,)."""
    only = np.ones(1, bool)
    agent.start(xs[0][None, :], only)
    weights = []
    for t, r in enumerate(rewards):
        done = t == len(rewards) - 1
        nxt = np.zeros((1, len(xs[0]))) if done else xs[t + 1][None, :]
        agent.step(r, nxt, only, done)
        weights.append(agent.w.copy())
    return weights


def online_lambda_return(xs, rewards, alpha, gamma, lam, w0):
    """The online lambda-return algorithm by brute force: every horizon h redoes all updates t < h."""
    T = len(rewards)
    d = len(w0)
    X = [np.asarray(x, float) for x in xs] + [np.zeros(d)]  # terminal features are zero
    diag = [np.asarray(w0, float).copy()]  # diag[h] = final weights of horizon h

    def nstep(t, n):
        g = sum(gamma**i * rewards[t + i] for i in range(n))
        return g + gamma**n * (diag[t + n - 1] @ X[t + n])

    for h in range(1, T + 1):
        w = np.asarray(w0, float).copy()
        for t in range(h):
            k = h - t
            g = (1 - lam) * sum(lam ** (n - 1) * nstep(t, n) for n in range(1, k)) + lam ** (k - 1) * nstep(t, k)
            w = w + alpha * (g - w @ X[t]) * X[t]
        diag.append(w)
    return diag[1:]


def random_walk(n=5):
    """Episodic random walk on n states; reward 1 on the right exit. Exact values by a linear solve."""
    P = np.zeros((n, n))
    r = np.zeros(n)
    for i in range(n):
        if i > 0:
            P[i, i - 1] = 0.5
        if i < n - 1:
            P[i, i + 1] = 0.5
        else:
            r[i] = 0.5
    return np.linalg.solve(np.eye(n) - P, r)


def walk_episode(rng, n):
    s, states, rewards = n // 2, [], []
    while True:
        states.append(s)
        s += 1 if rng.random() < 0.5 else -1
        if s < 0:
            rewards.append(0.0)
            return states, rewards
        if s >= n:
            rewards.append(1.0)
            return states, rewards
        rewards.append(0.0)


def learn_random_walk(agent, n, episodes, seed):
    """Tabular (one-hot) features; RMSE to the exact values of the weights averaged over the last half.

    With a constant step size the last weights keep a noise proportional to the step size (RMSE up
    to ~0.06 at alpha = 0.01, lambda = 1), so convergence is measured on the average.
    """
    rng = np.random.default_rng(seed)
    eye = np.eye(n)
    total = np.zeros(n)
    for ep in range(episodes):
        states, rewards = walk_episode(rng, n)
        run_fixed_episode(agent, [eye[s] for s in states], rewards)
        if ep >= episodes // 2:
            total += agent.w
    return float(np.sqrt(np.mean((total / (episodes - episodes // 2) - random_walk(n)) ** 2)))


# Control problem: corridor of 3 states; action 0 moves right (-1 per step), action 1 stays (-2).
# In state 1, action 1 is masked. Leaving state 2 to the right ends the episode with +10.
N_STATES, N_ACTIONS = 3, 2
MASK = np.array([[True, True], [True, False], [True, True]])


def corridor_phi(s):
    phi = np.zeros((N_ACTIONS, N_STATES * N_ACTIONS))
    phi[0, s * N_ACTIONS] = 1.0
    phi[1, s * N_ACTIONS + 1] = 1.0
    return phi


def corridor_step(s, a):
    if a == 1:
        return s, -2.0, False
    return (s + 1, -1.0, False) if s < N_STATES - 1 else (s, 9.0, True)


def corridor_q_star():
    """Value iteration on the corridor (discount 1)."""
    Q = np.zeros((N_STATES, N_ACTIONS))
    for _ in range(100):
        new = Q.copy()
        for s in range(N_STATES):
            for a in np.flatnonzero(MASK[s]):
                nxt, r, done = corridor_step(s, int(a))
                new[s, a] = r + (0.0 if done else Q[nxt][MASK[nxt]].max())
        Q = new
    return Q


def corridor_q_eps(eps):
    """Exact Q of the epsilon-greedy policy around the action 'move right' (what SARSA converges to)."""
    keys = [(s, int(a)) for s in range(N_STATES) for a in np.flatnonzero(MASK[s])]
    idx = {k: i for i, k in enumerate(keys)}
    A, b = np.eye(len(keys)), np.zeros(len(keys))
    for (s, a), i in idx.items():
        nxt, r, done = corridor_step(s, a)
        b[i] = r
        if not done:
            allowed = np.flatnonzero(MASK[nxt])
            for a2 in allowed:
                pi = eps / len(allowed) + ((1 - eps) if a2 == 0 else 0.0)
                A[i, idx[(nxt, int(a2))]] -= pi
    q = np.linalg.solve(A, b)
    return {k: q[i] for k, i in idx.items()}


def run_corridor(agent, episodes, max_steps=50):
    for _ in range(episodes):
        s = 0
        a = agent.start(corridor_phi(s), MASK[s])
        for t in range(max_steps):
            nxt, r, done = corridor_step(s, a)
            done = done or t == max_steps - 1
            a = agent.step(r, corridor_phi(nxt), MASK[nxt], done)
            s = nxt
            if done:
                break


def divergent_chain(agent, max_episodes=2000):
    """Tsitsiklis-Van Roy style: one update on s1 (feature 1) towards s2 (feature 2), reward 0.

    Bootstrapping off-policy, w grows by a factor 1 + alpha (2 gamma - 1 - eta) per update.
    """
    only = np.ones(1, bool)
    norms = []
    for _ in range(max_episodes):
        agent.start(np.array([[1.0]]), only)
        agent.step(0.0, np.array([[2.0]]), only, False)
        norms.append(float(np.linalg.norm(agent.w)))
        if agent.diverged:
            break
    return norms


# --- True online SARSA(lambda) ---------------------------------------------------------------


def test_one_hand_computed_update():
    a = TrueOnlineSarsaAgent(d=2, alpha=0.5, lam=0.5, gamma=1.0, rng=np.random.default_rng(0))
    a.w[:] = [1.0, 2.0]
    x0, x1 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    only = np.ones(1, bool)
    a.start(x0[None, :], only)
    a.step(1.0, x1[None, :], only, False)
    # Q = 1, Q' = 2, delta = 1 + 2 - 1 = 2, z = x0 (first step), Q_old = 0
    # w <- w + 0.5 (2 + 1 - 0) z - 0.5 (1 - 0) x0 = [1, 2] + [1.5, 0] - [0.5, 0]
    assert np.allclose(a.w, [2.0, 2.0])
    a.step(0.0, np.zeros((1, 2)), only, True)
    # Q = w.x1 = 2, Q' = 0, delta = -2, z = 0.5 x0 + (1 - 0.5*0.5*0) x1 = [0.5, 1], Q_old = 2 (old Q')
    # w <- [2, 2] + 0.5 (-2 + 2 - 2) [0.5, 1] - 0.5 (2 - 2) x1 = [2, 2] - [0.5, 1]
    assert np.allclose(a.w, [1.5, 1.0])


@pytest.mark.parametrize("lam", [0.0, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("gamma", [1.0, 0.9])
def test_weights_equal_the_online_lambda_return_algorithm(lam, gamma):
    rng = np.random.default_rng(int(lam * 10) + int(gamma * 100))
    T, d = 8, 4
    xs = list(rng.normal(size=(T, d)) * 0.5)
    rewards = list(rng.normal(size=T))
    w0 = rng.normal(size=d)
    agent = TrueOnlineSarsaAgent(d=d, alpha=0.1, lam=lam, gamma=gamma, rng=np.random.default_rng(0))
    agent.w[:] = w0
    got = run_fixed_episode(agent, xs, rewards)
    want = online_lambda_return(xs, rewards, 0.1, gamma, lam, w0)
    for g, w in zip(got, want):
        assert np.allclose(g, w, atol=1e-12)


def test_lambda_zero_equals_one_step_sarsa():
    rng = np.random.default_rng(1)
    d, alpha = 3, 0.2
    agent = TrueOnlineSarsaAgent(d=d, alpha=alpha, lam=0.0, gamma=0.95, rng=np.random.default_rng(0))
    w = np.zeros(d)
    for _ in range(5):  # traces and Q_old must reset between episodes
        xs = list(rng.normal(size=(6, d)))
        rewards = list(rng.normal(size=6))
        got = run_fixed_episode(agent, xs, rewards)
        for t in range(6):
            nxt = 0.0 if t == 5 else w @ xs[t + 1]
            w = w + alpha * (rewards[t] + 0.95 * nxt - w @ xs[t]) * xs[t]
            assert np.allclose(got[t], w, atol=1e-12)


@pytest.mark.parametrize("lam", [0.0, 0.5, 0.9, 1.0])
def test_converges_to_the_closed_form_values_of_a_markov_chain(lam):
    agent = TrueOnlineSarsaAgent(d=5, alpha=0.01, lam=lam, gamma=1.0, rng=np.random.default_rng(0))
    assert learn_random_walk(agent, 5, 8000, seed=3) < 0.02


def test_sarsa_control_converges_to_the_q_of_its_epsilon_greedy_policy():
    eps = 0.1
    agent = TrueOnlineSarsaAgent(d=N_STATES * N_ACTIONS, alpha=0.02, lam=0.5, gamma=1.0, rng=np.random.default_rng(0))
    agent.epsilon = eps
    run_corridor(agent, 6000)
    for (s, a), q in corridor_q_eps(eps).items():
        assert abs(agent.w[s * N_ACTIONS + a] - q) < 0.25, (s, a)


# --- RegQ ------------------------------------------------------------------------------------


def test_regq_update_by_hand():
    a = RegQAgent(d=2, alpha=0.5, eta=0.1, gamma=1.0, rng=np.random.default_rng(0))
    a.w[:] = [1.0, 2.0]
    only = np.ones(1, bool)
    phi_next = np.array([[0.0, 1.0], [1.0, 0.0]])  # max over the mask: action 1 is masked below
    a.start(np.array([[1.0, 0.0]]), only)
    a.step(1.0, phi_next, np.array([False, True]), False)
    # target = 1 + w.[1, 0] = 2, delta = 2 - 1 = 1, w <- w + 0.5 (1 * x - 0.1 w) = [1, 2] + [0.5, 0] - [0.05, 0.1]
    assert np.allclose(a.w, [1.45, 1.9])


def test_regq_without_regularization_finds_q_star():
    agent = RegQAgent(d=N_STATES * N_ACTIONS, alpha=0.1, eta=0.0, gamma=1.0, rng=np.random.default_rng(0))
    agent.epsilon = 0.3
    run_corridor(agent, 3000)
    q = corridor_q_star()
    for s in range(N_STATES):
        for a in np.flatnonzero(MASK[s]):
            assert abs(agent.w[s * N_ACTIONS + a] - q[s, a]) < 0.05, (s, a)


@pytest.mark.parametrize("eta", [0.0, 1e-4, 1e-3])
def test_divergence_check_fires_on_a_divergent_chain(eta):
    agent = RegQAgent(d=1, alpha=0.1, eta=eta, gamma=1.0, rng=np.random.default_rng(0))
    agent.w[:] = 1.0
    norms = divergent_chain(agent)
    assert agent.diverged and np.isfinite(agent.w).all()
    assert np.linalg.norm(agent.w) <= agent.weight_limit  # the last finite weights are kept
    before = agent.w.copy()
    agent.start(np.array([[1.0]]), np.ones(1, bool))
    agent.step(0.0, np.array([[2.0]]), np.ones(1, bool), False)
    assert np.array_equal(agent.w, before) and len(norms) < 2000  # learning has stopped


def test_large_regularization_stabilizes_the_divergent_chain():
    agent = RegQAgent(d=1, alpha=0.1, eta=1.5, gamma=1.0, rng=np.random.default_rng(0))
    agent.w[:] = 1.0
    divergent_chain(agent)
    assert not agent.diverged and abs(agent.w[0]) < 1e-6  # factor 1 + 0.1 (2 - 1 - 1.5) < 1


def test_nan_triggers_the_divergence_check():
    agent = RegQAgent(d=1, alpha=0.1, eta=0.0, gamma=1.0, rng=np.random.default_rng(0))
    agent.start(np.array([[1.0]]), np.ones(1, bool))
    agent.step(float("nan"), np.array([[1.0]]), np.ones(1, bool), False)
    assert agent.diverged and np.isfinite(agent.w).all()


# --- Myopic ----------------------------------------------------------------------------------


def test_myopic_regresses_on_the_immediate_target_without_bootstrap():
    a = MyopicAgent(d=2, alpha=0.5, rng=np.random.default_rng(0))
    a.w[:] = [1.0, 100.0]
    only = np.ones(1, bool)
    a.start(np.array([[1.0, 0.0]]), only)
    a.step(3.0, np.array([[0.0, 1.0]]), only, False)  # the large next value must not matter
    assert np.allclose(a.w, [2.0, 100.0])  # 1 + 0.5 (3 - 1)


def test_myopic_equals_sarsa_with_zero_gamma_and_lambda():
    rng = np.random.default_rng(4)
    xs, rewards = list(rng.normal(size=(10, 3))), list(rng.normal(size=10))
    m = run_fixed_episode(MyopicAgent(d=3, alpha=0.1, rng=np.random.default_rng(0)), xs, rewards)
    s = run_fixed_episode(TrueOnlineSarsaAgent(d=3, alpha=0.1, lam=0.0, gamma=0.0, rng=np.random.default_rng(0)), xs, rewards)
    for a, b in zip(m, s):
        assert np.allclose(a, b, atol=1e-12)


# --- Action selection and masks ----------------------------------------------------------------


def all_agents(d, seed=0):
    rng = lambda: np.random.default_rng(seed)  # noqa: E731
    return [TrueOnlineSarsaAgent(d=d, alpha=0.05, lam=0.5, gamma=1.0, rng=rng()),
            RegQAgent(d=d, alpha=0.05, eta=1e-3, gamma=1.0, rng=rng()), MyopicAgent(d=d, alpha=0.05, rng=rng())]


def test_masked_action_is_never_selected():
    rng = np.random.default_rng(0)
    phi = rng.normal(size=(8, 4))
    mask = np.array([0, 1, 0, 1, 1, 0, 0, 1], bool)
    for agent in all_agents(4):
        agent.w[:] = rng.normal(size=4) * 10  # the best unmasked and masked actions differ
        for eps in (0.0, 0.5, 1.0):
            agent.epsilon = eps
            assert all(mask[agent.select(phi, mask)] for _ in range(500))
        assert mask[agent.greedy(phi, mask)]


def test_empty_mask_gives_no_action():
    for agent in all_agents(2):
        assert agent.select(np.zeros((3, 2)), np.zeros(3, bool)) is None
        assert agent.greedy(np.zeros((3, 2)), np.zeros(3, bool)) is None


def test_greedy_picks_the_best_and_consumes_no_randomness():
    agent = all_agents(2)[0]
    phi = np.array([[1.0, 0.0], [0.0, 1.0], [5.0, 5.0]])
    mask = np.array([True, True, False])
    agent.w[:] = [1.0, 2.0]
    state = agent.rng.bit_generator.state
    assert agent.greedy(phi, mask) == 1  # action 2 is better but masked
    assert agent.rng.bit_generator.state == state


def test_exploration_is_uniform_over_unmasked_actions():
    agent = all_agents(1)[0]
    agent.epsilon = 1.0
    mask = np.array([True, False, True, True])
    counts = np.bincount([agent.select(np.zeros((4, 1)), mask) for _ in range(6000)], minlength=4)
    assert counts[1] == 0 and np.all(np.abs(counts[mask] - 2000) < 200)


def test_epsilon_schedules():
    s = EpsilonSchedule.linear(0.2, 0.01, frac=0.5, budget=100)
    assert s(0) == 0.2 and s(50) == 0.01 and s(99) == 0.01 and np.isclose(s(25), 0.105)
    assert EpsilonSchedule.constant(0.05)(12345) == 0.05


# --- On the environment: determinism, bounded weights, seeds -----------------------------------


def env_features(obs, env):
    """Small part-level features for these tests; the real ones come in Slice 2."""
    nom = env.nominal
    gain = (nom.visibility & ~obs["covered"].astype(bool)).sum(1) / nom.visibility.shape[1]
    return np.c_[np.ones_like(gain), gain, nom.scan_time / 10.0, np.full_like(gain, obs["step"][0] / env.k_max)]


def run_on_env(agent, env, episodes, seed=0):
    """Shaped-free training loop on the synthetic environment; rewards scaled as in Plan 002."""
    trace = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep, options={"variant": env.variants[0]})
        a = agent.start(env_features(obs, env), env.action_masks())
        done = False
        while not done:
            assert env.action_masks()[a]
            obs, r, term, trunc, _ = env.step(a)
            done = term or trunc
            trace.append(a)
            a = agent.step(0.01 * r, env_features(obs, env), env.action_masks(), done)
            assert (a is None) == done
    return trace


def test_same_seeds_same_run_and_other_seed_other_run():
    def go(seed):
        agent = TrueOnlineSarsaAgent(d=4, alpha=0.05, lam=0.9, gamma=1.0, rng=agents.agent_rng(seed))
        agent.epsilon = 0.3
        return run_on_env(agent, make_env(), 40), agent.w.copy()

    t1, w1 = go(0)
    t2, w2 = go(0)
    t3, _ = go(1)
    assert t1 == t2 and np.array_equal(w1, w2) and t1 != t3


@pytest.mark.parametrize("make", [
    lambda rng: TrueOnlineSarsaAgent(d=4, alpha=0.05, lam=0.9, gamma=1.0, rng=rng),
    lambda rng: RegQAgent(d=4, alpha=0.05, eta=1e-3, gamma=1.0, rng=rng),
    lambda rng: MyopicAgent(d=4, alpha=0.05, rng=rng),
])
def test_weights_stay_finite_and_bounded_on_the_environment(make):
    agent = make(np.random.default_rng(0))
    agent.epsilon = 0.2
    run_on_env(agent, make_env(), 200)
    assert np.isfinite(agent.w).all() and not agent.diverged and np.linalg.norm(agent.w) < 100


def test_tuning_and_official_seeds_do_not_overlap():
    assert not set(agents.TUNING_SEEDS) & set(agents.OFFICIAL_SEEDS)
    assert list(agents.OFFICIAL_SEEDS) == list(range(10)) and list(agents.TUNING_SEEDS) == [1000, 1001, 1002]


def test_agent_and_environment_seed_streams_are_separate():
    a, e = agents.agent_rng(3), agents.env_seeds(3, 5)
    assert not np.array_equal(a.integers(0, 2**31, 5), e)
    assert np.array_equal(agents.env_seeds(3, 5), e) and not np.array_equal(agents.env_seeds(4, 5), e)


def test_agents_are_numpy_only_and_share_the_interface():
    for agent in all_agents(2):
        assert isinstance(agent, LinearAgent)
    import nbv.agents.linear as m1
    import nbv.agents.myopic as m2
    import nbv.agents.qlearning as m3
    import nbv.agents.sarsa as m4
    import inspect

    for m in (m1, m2, m3, m4):
        src = inspect.getsource(m)
        for banned in ("import torch", "stable_baselines3", "sb3_contrib", "gymnasium"):
            assert banned not in src, (m.__name__, banned)
