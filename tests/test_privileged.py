"""Agent-side code never reads privileged data: the real variant (`env.data.real`, `RealPart`,
`env.coverages()`, `env.state()`) or `info["eval"]` (D-029, D-038).

Static: an AST search over features, baselines, shaping and agents; the only allowed use is
`greedy_oracle_policy` (upper reference by design). Runtime: nominal baselines and planners run
with the real part replaced by an object that fails on any access.
"""

import ast
import dataclasses
import importlib
import inspect
import pkgutil

import numpy as np
import pytest

import nbv.agents
from nbv import baselines, features, shaping
from test_env import make_env

ALLOWED = {"greedy_oracle_policy"}
FORBIDDEN_ATTRS = {"real", "coverages", "state"}


def agent_side_modules():
    mods = [features, baselines, shaping, nbv.agents]
    mods += [importlib.import_module(m.name) for m in pkgutil.walk_packages(nbv.agents.__path__, "nbv.agents.")]
    return mods


def privileged_uses(source: str, name: str) -> list[str]:
    tree = ast.parse(source)
    found = []

    def visit(node, func):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func = node.name
        hit = None
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            hit = f".{node.attr}"
        elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and node.slice.value == "eval":
            hit = '["eval"]'
        elif isinstance(node, ast.Name) and node.id == "RealPart":
            hit = "RealPart"
        if hit and func not in ALLOWED:
            found.append(f"{name}:{node.lineno} {hit} in {func}")
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, ast.ImportFrom):
                visit(child, func)

    visit(tree, "<module>")
    return found


def test_static_no_privileged_access_in_agent_side_code():
    found = [u for m in agent_side_modules() for u in privileged_uses(inspect.getsource(m), m.__name__)]
    assert not found, found


def test_static_check_detects_violations():
    src = "def f(env, info):\n    return env.data.real, info['eval'], env.coverages()\n"
    assert len(privileged_uses(src, "fake")) == 3
    assert privileged_uses("def greedy_oracle_policy(real: RealPart):\n    return real.union\n", "fake") == []


class _Poison:
    def __getattr__(self, name):
        raise AssertionError(f"privileged data accessed: real.{name}")


def poisoned_env():
    e = make_env()
    obs, _ = e.reset(options={"variant": e.variants[0]})
    e.data = dataclasses.replace(e.data, real=_Poison())
    return e, obs


@pytest.mark.parametrize("method", baselines.NOMINAL_POLICIES)
def test_runtime_nominal_policies_never_touch_the_real_part(method):
    e, obs = poisoned_env()
    rng = np.random.default_rng(0)
    policy = baselines.make_policy(method, e, rng)
    for covered in (obs["covered"], (np.arange(len(obs["covered"])) % 2).astype(np.int8)):
        a = policy(dict(obs, covered=covered), e.nominal, e.action_masks())
        assert a is None or e.action_masks()[a]


def test_runtime_planners_and_repair_never_touch_the_real_part():
    e, obs = poisoned_env()
    nom = e.nominal
    plans = [baselines.fixed_pattern(nom), baselines.ilp_tsp_plan(nom, time_limit_s=10)[0],
             baselines.random_tsp_plan(nom, 3, np.random.default_rng(0))[0],
             baselines.joint_opt_plan(nom, time_limit_s=10, workers=1)[0]]
    assert all(len(p) > 0 for p in plans)
    assert baselines.repair_policy(e.v, e.a)(obs, nom, e.action_masks()) is not None


def test_greedy_oracle_is_the_only_privileged_baseline():
    e, _ = poisoned_env()
    oracle = baselines.greedy_oracle_policy(e.data.real)
    obs = {"covered": np.zeros(e.nominal.visibility.shape[1], np.int8)}
    with pytest.raises(AssertionError):
        oracle(obs, e.nominal, e.action_masks())
