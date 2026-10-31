"""Baselines: random, fixed pattern, greedy-IG variants, greedy-oracle, set-cover ILP + TSP, random + TSP, MCTS.

Information per baseline (D-029, D-033): everything uses the nominal model except greedy-oracle,
which reads the real variant's visibility (upper reference). Plan-ahead methods are open loop and
are evaluated with plan repair (D-032). Closed-loop policies have the signature
(obs, nominal, mask) -> pass | None and run through `nbv.env.evaluate_policy`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from ortools.sat.python import cp_model

from nbv import features
from nbv.env import NominalModel, RealPart
from nbv.robot import movej_time, wrap_pi

Policy = Callable[[dict, NominalModel, np.ndarray], "int | None"]


# --- Closed-loop policies ------------------------------------------------------


def random_policy(rng: np.random.Generator) -> Policy:
    return lambda obs, nominal, mask: int(rng.choice(np.flatnonzero(mask)))


def greedy_ig_action(obs: dict, nominal: NominalModel, mask: np.ndarray) -> int | None:
    """Greedy-IG: max predicted new samples (nominal visibility)."""
    gain = np.where(mask, features.nominal_gain(obs, nominal), 0)
    return int(np.argmax(gain)) if gain.max() > 0 else None


def greedy_cost_action(obs: dict, nominal: NominalModel, mask: np.ndarray, v: float, a: float) -> int | None:
    """Greedy-IG/cost on the nominal model: max predicted new samples per second (move estimate + scan).

    Returns None when no unmasked pass is predicted to add a sample.
    """
    gain = features.nominal_gain(obs, nominal).astype(float)
    gain[~mask] = 0.0
    if gain.max() <= 0:
        return None
    t = features.move_time_estimate(obs, nominal, v, a) + nominal.scan_time
    score = np.where(mask & (gain > 0), gain / t, -np.inf)
    return int(np.argmax(score))


def greedy_cost_policy(v: float, a: float) -> Policy:
    return lambda obs, nominal, mask: greedy_cost_action(obs, nominal, mask, v, a)


def repair_policy(v: float, a: float) -> Policy:
    """Plan repair for open-loop methods (D-032): cost-aware greedy with nominal visibility."""
    return greedy_cost_policy(v, a)


def greedy_oracle_policy(real: RealPart) -> Policy:
    """Greedy-oracle: max new samples of the real variant (privileged; upper reference only)."""

    def act(obs: dict, nominal: NominalModel, mask: np.ndarray) -> int | None:
        new = real.union & ~obs["covered"].astype(bool)
        gain = np.where(mask, (real.visibility & new).sum(1), 0)
        return int(np.argmax(gain)) if gain.max() > 0 else None

    return act


# --- Fixed pattern -------------------------------------------------------------


def fixed_pattern(nominal: NominalModel, ring_elevation_deg: float = 45.0, ring_size: int = 8) -> list[int]:
    """Top pass + ring at `ring_elevation_deg`, snapped to the nearest reachable candidate (D-033).

    Robot analogue of stepping a turntable in front of a fixed scanner. Duplicates are skipped;
    order: top, then the ring by azimuth starting at 0 deg.
    """
    el = np.radians(ring_elevation_deg)
    az = 2 * np.pi * np.arange(ring_size) / ring_size
    targets = np.vstack([[0.0, 0.0, 1.0], np.c_[np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.full(ring_size, np.sin(el))]])
    cand = np.flatnonzero(nominal.reachable)
    out: list[int] = []
    for t in targets:
        j = int(cand[np.argmax(nominal.direction[cand] @ t)])
        if j not in out:
            out.append(j)
    return out


# --- Set cover / max coverage (CP-SAT) -------------------------------------------


@dataclass
class SolverResult:
    passes: list[int]
    value: float
    status: str
    gap: float
    wall_s: float
    extra: dict = field(default_factory=dict)


def _element_classes(vis: np.ndarray, ref: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Group reference samples with identical coverage columns: (classes x rows) bool, weights."""
    cols = vis[np.ix_(rows, np.flatnonzero(ref))].T  # (samples, rows)
    uniq, counts = np.unique(np.packbits(cols, axis=1), axis=0, return_counts=True)
    return np.unpackbits(uniq, axis=1, count=len(rows)).astype(bool), counts


def _solve(model: cp_model.CpModel, time_limit_s: float, workers: int = 8) -> tuple[cp_model.CpSolver, str]:
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_workers = workers
    solver.parameters.random_seed = 0
    status = solver.Solve(model)
    return solver, solver.StatusName(status)


def set_cover(vis: np.ndarray, ref: np.ndarray, rows: np.ndarray, fraction: float, time_limit_s: float = 60.0) -> SolverResult:
    """Fewest passes (among `rows`) covering >= `fraction` of the `ref` samples (partial set cover).

    Two stages so the result is both optimal and reproducible (D-033): (1) minimize the number of
    passes; (2) with that number fixed, minimize a tie-break with distinct seeded weights per pass,
    whose optimum is unique. With several workers, a single-stage solve returned a different optimal
    cover on repeated runs; with one worker, the 95% cover at 128 candidates was not solved to
    optimality within the time limit.
    """
    cls, w = _element_classes(vis, ref, rows)
    need = math.ceil(fraction * w.sum() - 1e-9)

    def model(count: int | None):
        m = cp_model.CpModel()
        x = [m.NewBoolVar(f"x{i}") for i in range(len(rows))]
        if need >= w.sum():
            for c in cls:
                m.AddBoolOr([x[i] for i in np.flatnonzero(c)])
        else:
            y = [m.NewBoolVar(f"y{e}") for e in range(len(cls))]
            for e, c in enumerate(cls):
                m.AddBoolOr([x[i] for i in np.flatnonzero(c)]).OnlyEnforceIf(y[e])
            m.Add(sum(int(w[e]) * y[e] for e in range(len(cls))) >= need)
        if count is None:
            m.Minimize(sum(x))
        else:
            m.Add(sum(x) == count)
            tie = np.random.default_rng(0).permutation(len(rows)) + 1  # distinct weights -> unique optimum
            m.Minimize(sum(int(tie[i]) * x[i] for i in range(len(rows))))
        return m, x

    m1, _ = model(None)
    s1, status = _solve(m1, time_limit_s)
    obj, bound = s1.ObjectiveValue(), s1.BestObjectiveBound()
    m2, x = model(int(round(obj)))
    s2, status2 = _solve(m2, time_limit_s)
    chosen = [int(rows[i]) for i in range(len(rows)) if s2.Value(x[i])]
    return SolverResult(chosen, obj, status, (obj - bound) / max(obj, 1), s1.WallTime() + s2.WallTime(),
                        {"n_classes": len(cls), "tie_break_status": status2})


def max_coverage(vis: np.ndarray, ref: np.ndarray, rows: np.ndarray, k: int, time_limit_s: float = 60.0) -> SolverResult:
    """Largest coverage of `ref` with at most k passes (exact upper bound for any k-pass sequence)."""
    cls, w = _element_classes(vis, ref, rows)
    m = cp_model.CpModel()
    x = [m.NewBoolVar(f"x{i}") for i in range(len(rows))]
    y = [m.NewBoolVar(f"y{e}") for e in range(len(cls))]
    for e, c in enumerate(cls):
        m.AddBoolOr([x[i] for i in np.flatnonzero(c)]).OnlyEnforceIf(y[e])
    m.Add(sum(x) <= k)
    m.Maximize(sum(int(w[e]) * y[e] for e in range(len(cls))))
    solver, status = _solve(m, time_limit_s)
    chosen = [int(rows[i]) for i in range(len(rows)) if solver.Value(x[i])]
    total = w.sum()
    obj, bound = solver.ObjectiveValue(), solver.BestObjectiveBound()
    # Report the bound, not the incumbent, so the value stays an upper bound if the time limit hits.
    return SolverResult(chosen, bound / total, status, (bound - obj) / max(bound, 1), solver.WallTime(), {"incumbent": obj / total})


# --- TSP -------------------------------------------------------------------------


def _tour_costs(nominal: NominalModel, passes: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.asarray(passes)
    return nominal.from_home_pi[p], nominal.ptp_pi[np.ix_(p, p)], nominal.to_home_pi[p]


def held_karp(start: np.ndarray, C: np.ndarray, end: np.ndarray) -> tuple[list[int], float]:
    """Exact asymmetric path home -> all nodes -> home. Returns the order (node indices) and its cost."""
    n = len(start)
    full = (1 << n) - 1
    dp = np.full((1 << n, n), np.inf)
    parent = np.full((1 << n, n), -1, dtype=np.int64)
    for j in range(n):
        dp[1 << j, j] = start[j]
    for mask in range(1, 1 << n):
        row = dp[mask]
        if not np.isfinite(row).any():
            continue
        cand = row[:, None] + C  # (from j, to k)
        best_j = np.argmin(cand, axis=0)
        best = cand[best_j, np.arange(n)]
        ks = np.flatnonzero(((mask >> np.arange(n)) & 1) == 0)
        nm = mask | (1 << ks)
        better = best[ks] < dp[nm, ks]
        dp[nm[better], ks[better]] = best[ks[better]]
        parent[nm[better], ks[better]] = best_j[ks[better]]
    total = dp[full] + end
    last = int(np.argmin(total))
    order, mask = [], full
    while last >= 0:
        order.append(last)
        prev = int(parent[mask, last])
        mask ^= 1 << last
        last = prev
    return order[::-1], float(total.min())


def routing_tsp(start: np.ndarray, C: np.ndarray, end: np.ndarray, time_limit_s: float = 10.0) -> tuple[list[int], float]:
    """OR-Tools routing with the home as depot (node 0); costs in ms."""
    n = len(start)
    M = np.zeros((n + 1, n + 1))
    M[0, 1:], M[1:, 0], M[1:, 1:] = start, end, C
    M = np.rint(M * 1000).astype(np.int64)
    manager = pywrapcp.RoutingIndexManager(n + 1, 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    cb = routing.RegisterTransitCallback(lambda i, j: int(M[manager.IndexToNode(i), manager.IndexToNode(j)]))
    routing.SetArcCostEvaluatorOfAllVehicles(cb)
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromMilliseconds(int(time_limit_s * 1000))
    sol = routing.SolveWithParameters(params)
    order, idx = [], routing.Start(0)
    while not routing.IsEnd(idx):
        node = manager.IndexToNode(idx)
        if node:
            order.append(node - 1)
        idx = sol.Value(routing.NextVar(idx))
    return order, sol.ObjectiveValue() / 1000.0


def tsp_order(nominal: NominalModel, passes: list[int], exact_max: int = 16, time_limit_s: float = 10.0) -> tuple[list[int], dict]:
    """Order `passes` to minimize planner travel time (ptp_pi, home -> ... -> home), D-033."""
    if len(passes) <= 1:
        return list(passes), {"tsp": "trivial"}
    s, C, e = _tour_costs(nominal, passes)
    if len(passes) <= exact_max:
        order, cost = held_karp(s, C, e)
        how = "held_karp"
    else:
        order, cost = routing_tsp(s, C, e, time_limit_s)
        how = "ortools_routing"
    return [passes[i] for i in order], {"tsp": how, "tsp_cost_s": cost}


def tour_cost(nominal: NominalModel, seq: list[int]) -> float:
    s, C, e = _tour_costs(nominal, seq)
    return float(s[0] + sum(C[i, i + 1] for i in range(len(seq) - 1)) + e[-1])


# --- Plan-ahead baselines ----------------------------------------------------------


def ilp_tsp_plan(nominal: NominalModel, fraction: float = 0.95, time_limit_s: float = 60.0) -> tuple[list[int], dict]:
    """Industry-type CAD-based planner (D-033): set cover on the nominal CAD, then TSP."""
    rows = np.flatnonzero(nominal.reachable)
    sc = set_cover(nominal.visibility, nominal.union, rows, fraction, time_limit_s)
    order, info = tsp_order(nominal, sc.passes)
    return order, {"ilp_status": sc.status, "ilp_gap": sc.gap, "ilp_passes": len(sc.passes), "ilp_wall_s": sc.wall_s, **info}


def planner_cost(nominal: NominalModel, seq: list[int]) -> float:
    """Planner objective of a pass sequence: scan times + ptp_pi tour from home back to home (s)."""
    return tour_cost(nominal, seq) + float(nominal.scan_time[seq].sum()) if seq else 0.0


def joint_opt_plan(nominal: NominalModel, fraction: float = 0.95, time_limit_s: float = 300.0,
                   hint: list[int] | None = None, workers: int = 8) -> tuple[list[int], dict]:
    """Joint optimum reference (D-035): choose and order passes in one CP-SAT model.

    Nodes: home (0) + reachable passes. A circuit through home; skipped passes take their self-loop.
    Coverage: >= `fraction` of the nominal reachable union (merged sample classes, as in set cover).
    Objective: scan times of chosen passes + ptp_pi arc times incl. leaving and returning home, in ms.
    `hint` (e.g. the ILP + TSP plan) is given as a solution hint. Returns the plan and solver info,
    including CP-SAT's lower bound, so the gap to the true joint optimum is bracketed.
    """
    rows = np.flatnonzero(nominal.reachable)
    R = len(rows)
    ms = lambda t: int(round(float(t) * 1000))  # noqa: E731
    m = cp_model.CpModel()
    x = [m.NewBoolVar(f"x{i}") for i in range(R)]
    arcs, cost_terms, arc_var = [], [], {}
    for i in range(R):
        arcs.append((i + 1, i + 1, x[i].Not()))  # skipped pass
        a0, a1 = m.NewBoolVar(f"h_{i}"), m.NewBoolVar(f"{i}_h")
        arcs += [(0, i + 1, a0), (i + 1, 0, a1)]
        arc_var[(-1, i)], arc_var[(i, -1)] = a0, a1
        cost_terms += [ms(nominal.from_home_pi[rows[i]]) * a0, ms(nominal.to_home_pi[rows[i]]) * a1,
                       ms(nominal.scan_time[rows[i]]) * x[i]]
        for j in range(R):
            if i != j:
                b = m.NewBoolVar(f"{i}_{j}")
                arcs.append((i + 1, j + 1, b))
                arc_var[(i, j)] = b
                cost_terms.append(ms(nominal.ptp_pi[rows[i], rows[j]]) * b)
    m.AddCircuit(arcs)
    cls, w = _element_classes(nominal.visibility, nominal.union, rows)
    y = [m.NewBoolVar(f"y{e}") for e in range(len(cls))]
    for e, c in enumerate(cls):
        m.AddBoolOr([x[i] for i in np.flatnonzero(c)]).OnlyEnforceIf(y[e])
    m.Add(sum(int(w[e]) * y[e] for e in range(len(cls))) >= math.ceil(fraction * w.sum() - 1e-9))
    m.Minimize(sum(cost_terms))
    if hint:
        idx = {int(r): i for i, r in enumerate(rows)}
        h = [idx[p] for p in hint]
        on = set(zip([-1, *h], [*h, -1]))
        for key, var in arc_var.items():
            m.AddHint(var, key in on)
        for i in range(R):
            m.AddHint(x[i], i in h)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_workers = workers
    solver.parameters.random_seed = 0
    status = solver.StatusName(solver.Solve(m))
    # Follow the circuit from home.
    order, node = [], -1
    for _ in range(R):
        nxt = next((j for j in range(R) if (node, j) in arc_var and solver.Value(arc_var[(node, j)])), None)
        if nxt is None:
            break
        order.append(int(rows[nxt]))
        node = nxt
    obj, bound = solver.ObjectiveValue() / 1000, solver.BestObjectiveBound() / 1000
    return order, {"joint_status": status, "joint_obj_s": obj, "joint_bound_s": bound,
                   "joint_wall_s": solver.WallTime(), "joint_passes": len(order)}


def random_tsp_plan(nominal: NominalModel, k: int, rng: np.random.Generator) -> tuple[list[int], dict]:
    rows = np.flatnonzero(nominal.reachable)
    pick = [int(i) for i in rng.choice(rows, size=min(k, len(rows)), replace=False)]
    order, info = tsp_order(nominal, pick)
    return order, {"k": len(pick), **info}


# --- MCTS (nominal model) --------------------------------------------------------------


class MCTSPlanner:
    """UCT over the nominal model (D-033): each real step plans from the measured samples.

    Simulated state: samples covered so far (nominal union classes) and the last pass. Cost of a
    simulated pass: ptp_pi transfer (the first from the current joint state) + scan time; reaching
    the target adds the return home; failing within the remaining budget adds `fail_cost_s`.
    Expansion is limited to the `width` best passes by the cost-aware greedy score; rollouts are
    cost-aware greedy with epsilon-random choices. Value = -cost / scale_s. If the nominal model
    already predicts the target while the real coverage is below it, the cost-aware greedy pass is
    taken (as in plan repair, D-032).
    """

    def __init__(self, v: float, a: float, rng: np.random.Generator, k_max: int, target: float,
                 n_sim: int = 200, width: int = 8, c_uct: float = 1.0, epsilon: float = 0.2,
                 fail_cost_s: float = 250.0, scale_s: float = 100.0):
        self.v, self.a, self.rng, self.k_max, self.target = v, a, rng, k_max, target
        self.n_sim, self.width, self.c, self.eps = n_sim, width, c_uct, epsilon
        self.fail, self.scale = fail_cost_s, scale_s

    def __call__(self, obs: dict, nominal: NominalModel, mask: np.ndarray) -> int | None:
        cls, w = _element_classes(nominal.visibility, nominal.union, np.arange(len(mask)))
        V = cls.T  # (passes, classes)
        covered_samples = obs["covered"].astype(bool)[nominal.union]
        # A class counts as covered if its samples are (classes are exact groups of identical columns).
        cols = nominal.visibility[:, nominal.union].T
        key = {r.tobytes(): e for e, r in enumerate(np.packbits(cls, axis=1))}
        cls_of_sample = np.array([key[r.tobytes()] for r in np.packbits(cols, axis=1)])
        cov0 = np.zeros(len(w), bool)
        np.logical_or.at(cov0, cls_of_sample, covered_samples)
        need = self.target * w.sum()
        first_move = movej_time(wrap_pi(nominal.q_plan[:, 0] - obs["q"][None, :]), self.v, self.a)
        budget = self.k_max - int(obs["step"][0])
        if not mask.any():
            return None

        def step_cost(last: int | None, j: int) -> float:
            return float(first_move[j] if last is None else nominal.ptp_pi[last, j]) + float(nominal.scan_time[j])

        def candidates(cov, avail, last):
            gain = (V[:, ~cov] @ w[~cov]).astype(float)
            gain[~avail] = 0
            ok = np.flatnonzero(gain > 0)
            if len(ok) == 0:
                return ok, gain
            t = np.array([step_cost(last, j) for j in ok])
            return ok[np.argsort(-gain[ok] / t)][: self.width], gain

        def done(cov):
            return (w[cov]).sum() >= need - 1e-9

        def rollout(cov, avail, last, depth):
            cost = 0.0
            cov, avail = cov.copy(), avail.copy()
            while not done(cov):
                if depth >= budget:
                    return cost + self.fail
                ok, gain = candidates(cov, avail, last)
                if len(ok) == 0:
                    return cost + self.fail
                j = int(self.rng.choice(ok)) if self.rng.random() < self.eps else int(ok[0])
                cost += step_cost(last, j)
                cov |= V[j]
                avail[j] = False
                last, depth = j, depth + 1
            return cost + float(nominal.to_home_pi[last])

        if done(cov0):
            # The nominal model says the target is reached but the real coverage is not: same rule as plan repair.
            return greedy_cost_action(obs, nominal, mask, self.v, self.a)
        root = {"n": 0, "children": {}}
        root_ok, _ = candidates(cov0, mask.copy(), None)
        if len(root_ok) == 0:
            return None
        for _ in range(self.n_sim):
            node, cov, avail, last, depth, path, cost = root, cov0.copy(), mask.copy(), None, 0, [root], 0.0
            while True:
                if done(cov) or depth >= budget:
                    total = cost + (float(nominal.to_home_pi[last]) if done(cov) else self.fail)
                    break
                ok, _ = candidates(cov, avail, last)
                if len(ok) == 0:
                    total = cost + self.fail
                    break
                untried = [j for j in ok if j not in node["children"]]
                if untried:
                    j = int(untried[0])
                    child = {"n": 0, "q": 0.0, "children": {}}
                    node["children"][j] = child
                    cost += step_cost(last, j)
                    cov |= V[j]
                    avail[j] = False
                    path.append(child)
                    total = cost + rollout(cov, avail, j, depth + 1)
                    break
                logn = math.log(node["n"] + 1)
                j = max(ok, key=lambda j: node["children"][j]["q"] + self.c * math.sqrt(logn / (node["children"][j]["n"] + 1e-9)))
                j = int(j)
                cost += step_cost(last, j)
                cov |= V[j]
                avail[j] = False
                last, depth, node = j, depth + 1, node["children"][j]
                path.append(node)
            value = -total / self.scale
            for nd in path:
                nd["n"] += 1
                if "q" in nd:
                    nd["q"] += (value - nd["q"]) / nd["n"]
        return int(max(root["children"], key=lambda j: root["children"][j]["n"]))
