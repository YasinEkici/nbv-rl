"""Plan 002 training orchestration; the only learning loop is nbv.agents.train.train.

This module selects only train and VAL variants. It provides exploratory smoke/tuning runs
and a configurable entry point for later authorized official runs, never a test evaluation.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import itertools
import json
from pathlib import Path
import time

import numpy as np

from nbv import evaluation, parts, shaping
from nbv.agents import TUNING_SEEDS, agent_rng
from nbv.agents.linear import EpsilonSchedule
from nbv.agents.myopic import MyopicAgent
from nbv.agents.qlearning import RegQAgent
from nbv.agents.sarsa import TrueOnlineSarsaAgent
from nbv.agents.train import train
from nbv.env import ENV_VERSION, NBVEnv
from nbv.feature_check import CONFIG_FILES, validation_reference

METHODS = ("sarsa", "regq", "myopic")


class StopRun(RuntimeError):
    """Explicit user stop condition; completed records remain available."""


def grid_settings(method: str) -> list[dict]:
    if method not in METHODS:
        raise ValueError(method)
    key, values = {"sarsa": ("lam", (.5, .9)), "regq": ("eta", (.001, .01)),
                   "myopic": ("phi_alpha", (25., 50.))}[method]
    return [{"alpha": alpha, key: value, "epsilon": epsilon}
            for alpha, value, epsilon in itertools.product((.01, .03, .1), values, ("linear", "constant"))]


def make_agent(method: str, d: int, setting: dict, seed: int):
    # Only SARSA's grid explicitly divides alpha by d (Plan 002 section 2.4).
    kwargs = dict(d=d, alpha=setting["alpha"] / d if method == "sarsa" else setting["alpha"], rng=agent_rng(seed))
    if method == "sarsa":
        return TrueOnlineSarsaAgent(**kwargs, lam=setting.get("lam", .9), gamma=1.)
    if method == "regq":
        return RegQAgent(**kwargs, eta=setting.get("eta", .001), gamma=1.)
    if method == "myopic":
        return MyopicAgent(**kwargs)
    raise ValueError(method)


def select_setting(points: list[dict]) -> dict:
    if not points or any(len(p["scores"]) != 3 or not np.isfinite(p["scores"]).all() for p in points):
        raise StopRun("incomplete or nonfinite tuning scores")
    best = min(float(np.mean(p["scores"])) for p in points)
    tied = [p for p in points if float(np.mean(p["scores"])) <= best + .005]
    return min(tied, key=lambda p: (float(np.std(p["scores"])), p["setting"]["alpha"],
                                   json.dumps(p["setting"], sort_keys=True)))


def eligible_cache_paths(train_ids: list[str], val_ids: list[str]) -> list[str]:
    ids = sorted({vid.split("_")[0] for vid in train_ids + val_ids})
    paths = []
    for fid in ids:
        paths.append(f"data/cache/visibility/nominal/points_{fid}.npz")
        for n in (48, 128):
            paths += [f"data/cache/robot/nominal/{fid}_n{n}.npz", f"data/cache/visibility/nominal/{fid}_n{n}.npz"]
    paths += [f"data/cache/visibility/{vid}_n{n}.npz" for vid in train_ids + val_ids for n in (48, 128)]
    return paths


def context(status: str, registry: str, outputs: str) -> dict:
    manifest = parts.load_yaml("configs/split_v1.yaml")
    train_ids = [v["variant_id"] for v in manifest["variants"] if v["split"] == "train"]
    val_ids = [v["variant_id"] for v in manifest["variants"] if v["split"] == "val"]
    reference, reference_path = validation_reference(registry, outputs)
    if set(reference) != set(val_ids):
        raise StopRun("reference does not exactly cover VAL")
    git = evaluation.git_state()
    evaluation.check_status(status, git)
    return {"train": train_ids, "val": val_ids, "reference": reference,
            "provenance": git | {"env_version": ENV_VERSION, "split_version": "v1",
                "code_sha256": evaluation.code_hash(), "config_files": evaluation.file_hashes(CONFIG_FILES),
                "reference": evaluation.file_hashes([reference_path]),
                "caches": evaluation.selected_cache_manifest(eligible_cache_paths(train_ids, val_ids)),
                "train_variants": train_ids, "val_variants": val_ids}}


def run_one(task: dict) -> dict:
    method, setting, budget, d = (task[k] for k in ("method", "setting", "budget", "features"))
    wrapper = shaping.PotentialShaping(NBVEnv(task["train"]), phi=task["phi"],
        alpha=setting.get("phi_alpha", 50.), cache_value=task.get("cache_value", True))
    val_env, feature_cache, latest = NBVEnv(task["val"]), {}, []

    def assess(weights):
        nonlocal latest
        latest = evaluation.evaluate_weights(val_env, task["val"], weights, task["reference"],
                                             reduced=d == 11, feature_cache=feature_cache)
        summary = evaluation.aggregate(latest)
        if not np.isfinite(summary["normalized_cycle_time"]) or not np.isfinite(summary["success_rate"]):
            raise StopRun("nonfinite validation score")
        return summary

    agent = make_agent(method, d, setting, task["seed"])
    schedule = (EpsilonSchedule.linear(.2, .01, .5, budget) if setting["epsilon"] == "linear"
                else EpsilonSchedule.constant(.05))
    result = train(agent, wrapper, seed=task["seed"], budget=budget, eval_every=budget // 20,
                   schedule=schedule, evaluate=assess, reduced=d == 11, stop_on_nonfinite=True)
    if agent.divergence_reason == "nonfinite":
        raise StopRun("nonfinite weight update")
    if not all(np.isfinite(h["weight_norm"]) for h in result.history):
        raise StopRun("nonfinite weight norm")
    episodes = latest
    secondary = evaluation.evaluate_weights(val_env, task["val"], result.final_weights, task["reference"],
                                             reduced=d == 11, feature_cache=feature_cache)
    return {"task": {k: v for k, v in task.items() if k not in ("train", "val", "reference")},
            "summary": evaluation.aggregate(episodes), "episodes": episodes, "secondary": secondary,
            "history": result.history, "score": float(np.mean([h["normalized_cycle_time"] for h in result.history[-2:]])),
            "policy_weights": result.policy_weights.tolist(), "final_weights": result.final_weights.tolist(),
            "episode_seeds": result.episode_seeds, "tail_count": result.tail_count,
            "diverged": result.diverged, "divergence_reason": agent.divergence_reason,
            "wall_s": result.wall_s, "learning_s": result.learning_s, "evaluation_s": result.evaluation_s,
            "cache_hits": wrapper.cache_hits, "cache_misses": wrapper.cache_misses,
            "effective_alpha": agent.alpha}


def persist(result: dict, ctx: dict, status: str, tag: str, registry: str, outputs: str) -> dict:
    task = result["task"]
    params = task | {"effective_alpha": result["effective_alpha"], "reward_scale": .01, "gamma": 1.,
                     "evaluations": 20, "tail_fraction": .1}
    extra = {"training_run": True, "tag": tag, "n_candidates": 48, "feature_count": task["features"],
             "phi": task["phi"], "cache_value": task.get("cache_value", True), "setting": task["setting"],
             "grid_id": task.get("grid_id"), "selection_normalized_cycle_time": result["score"],
             **{k: result[k] for k in ("history", "tail_count", "diverged", "divergence_reason", "wall_s",
                                      "learning_s", "evaluation_s", "cache_hits", "cache_misses", "effective_alpha")},
             "steps_per_s": task["budget"] / result["learning_s"],
             "secondary_final_weights": evaluation.aggregate(result["secondary"])}
    run_id = evaluation.new_run_id(f"{tag}-{task['method']}-val-s{task['seed']}-{ENV_VERSION}")
    records = [e | {"method": task["method"], "seed": task["seed"], "split": "val", "run_id": run_id}
               for e in result["episodes"]]
    out = Path(outputs) / run_id
    evaluation.write_run(out, params, ctx["provenance"], records, result["summary"])
    (out / "training.json").write_text(json.dumps({k: result[k] for k in
        ("history", "policy_weights", "final_weights", "episode_seeds", "secondary", "tail_count")}, indent=1), encoding="utf-8")
    np.savez_compressed(out / "weights.npz", tail=np.asarray(result["policy_weights"]), final=np.asarray(result["final_weights"]))
    with open(out / "curve.csv", "w", newline="", encoding="utf-8") as stream:
        fields = ["step", "normalized_cycle_time", "success_rate", "weight_norm", "diverged"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: h[key] for key in fields} for h in result["history"])
    row = evaluation.registry_row(run_id, task["method"], evaluation.config_ref(ctx["provenance"]["config_files"], params),
        "val", task["seed"], result["summary"], extra, ENV_VERSION, "v1",
        evaluation.commit_label(ctx["provenance"]), status=status, plan="002", budget_steps=task["budget"])
    evaluation.append_registry(Path(registry), [row])
    print(f"{tag} {task['method']} grid={task.get('grid_id')} phi={task['phi']} seed={task['seed']} "
          f"score={result['score']:.6f} success={result['summary']['success_rate']:.3f} "
          f"diverged={result['diverged']} steps/s={extra['steps_per_s']:.1f}", flush=True)
    return row


def run_tasks(tasks: list[dict], *, status: str, tag: str, workers: int = 20,
              registry: str = "results/registry.csv", outputs: str = "outputs") -> list[dict]:
    if any(t["budget"] < 20 or t["budget"] % 20 or t["features"] not in (11, 16) for t in tasks):
        raise ValueError("budget must be divisible by 20 and features must be 11 or 16")
    ctx = context(status, registry, outputs)
    tasks = [t | {k: ctx[k] for k in ("train", "val", "reference")} for t in tasks]
    rows = []
    if workers == 1:
        for task in tasks:
            rows.append(persist(run_one(task), ctx, status, tag, registry, outputs))
        return rows
    pool = ProcessPoolExecutor(workers)
    try:
        futures = {pool.submit(run_one, t): t for t in tasks}
        for future in as_completed(futures):
            rows.append(persist(future.result(), ctx, status, tag, registry, outputs))
            if tag == "p002t":
                diverged = sum(r["method"] == "regq" and json.loads(r["extra_json"])["diverged"] for r in rows)
                if diverged > 18:
                    raise StopRun("RegQ softmax fallback triggered: more than 18 of 36 tuning runs diverged")
    except BaseException:
        pool.terminate_workers()  # Python 3.14: stop all in-flight work on the user's stop condition
        raise
    else:
        pool.shutdown()
    return rows


def tuning_tasks() -> list[dict]:
    return [{"method": method, "setting": setting, "seed": seed, "budget": 200000,
             "features": 11, "phi": "linear", "grid_id": f"{method}-{i:02d}"}
            for method in METHODS for i, setting in enumerate(grid_settings(method)) for seed in TUNING_SEEDS]


def smoke() -> list[dict]:
    tasks = [{"method": m, "setting": {"alpha": .03, "lam": .9, "eta": .001, "phi_alpha": 50., "epsilon": "linear"},
              "seed": 1000, "budget": 10000, "features": 11, "phi": phi, "cache_value": cache}
             for m in METHODS for phi, cache in [(p, True) for p in shaping.PHIS] + [("value_estimate", False)]]
    tasks.append(tasks[1] | {"features": 16})  # full-feature linear SARSA estimate, never an official run
    return run_tasks(tasks, status="exploratory", tag="p002smoke", workers=1)
