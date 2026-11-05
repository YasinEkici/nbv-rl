"""D-046 exploratory feature comparison and feature throughput measurement.

Orchestration only: learning is in nbv.agents.train, evaluation in nbv.evaluation.
All variant lists are explicitly restricted to train/val before constructing environments.
"""

from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
import time

import numpy as np

from nbv import evaluation, parts, shaping
from nbv.agents import TUNING_SEEDS, agent_rng
from nbv.agents.linear import EpsilonSchedule
from nbv.agents.sarsa import TrueOnlineSarsaAgent
from nbv.agents.train import train
from nbv.env import NBVEnv, ENV_VERSION
from nbv.features import ActionFeatures

CONFIG_FILES = ["configs/env_v4.yaml", "configs/robot_v2.yaml", "configs/collision_v1.yaml",
                "configs/parts_v2.yaml", "configs/split_v1.yaml", "configs/candidates_v1.yaml",
                "configs/sensor_v1.yaml", "configs/visibility_v1.yaml"]


def validation_reference(registry: str, outputs: str) -> tuple[dict[str, float], str]:
    rows = list(csv.DictReader(open(registry, encoding="utf-8")))
    matches = [r for r in rows if r["status"] == "canonical" and r["env_version"] == ENV_VERSION
               and r["plan"] == "001" and r["method"] == "ilp_tsp" and r["part_split"] == "val"
               and json.loads(r["extra_json"])["n_candidates"] == 48]
    if len(matches) != 1:
        raise ValueError("expected exactly one canonical val ILP+TSP reference")
    path = Path(outputs) / matches[0]["run_id"] / "episodes.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return {r["variant"]: r["cycle_time"] for r in records}, str(path)


def run_one(task: dict) -> dict:
    reduced, seed = task["features"] == 11, task["seed"]
    d, budget = task["features"], 200000
    training_env = shaping.PotentialShaping(NBVEnv(task["train"]), phi="linear", alpha=50.)
    validation_env = NBVEnv(task["val"])
    feature_cache, last_records = {}, []

    def assess(weights):
        nonlocal last_records
        last_records = evaluation.evaluate_weights(validation_env, task["val"], weights, task["reference"],
                                                   reduced=reduced, feature_cache=feature_cache)
        return evaluation.aggregate(last_records)

    agent = TrueOnlineSarsaAgent(d, .03 / d, .9, rng=agent_rng(seed))
    result = train(agent, training_env, seed=seed, budget=budget, eval_every=10000, reduced=reduced,
                   schedule=EpsilonSchedule.linear(.2, .01, .5, budget), evaluate=assess)
    final_records = last_records
    secondary = evaluation.evaluate_weights(validation_env, task["val"], result.final_weights, task["reference"],
                                            reduced=reduced, feature_cache=feature_cache)
    tail_evals = [h for h in result.history if h["step"] > 180000]
    score = float(np.mean([h["normalized_cycle_time"] for h in tail_evals]))
    return {"task": task, "score": score, "summary": evaluation.aggregate(final_records),
            "episodes": final_records, "secondary": secondary, "history": result.history,
            "policy_weights": result.policy_weights.tolist(), "final_weights": result.final_weights.tolist(),
            "tail_count": result.tail_count, "diverged": result.diverged, "wall_s": result.wall_s,
            "episode_seeds": result.episode_seeds}


def run_check(*, workers: int = 3, registry: str = "results/registry.csv", outputs: str = "outputs") -> None:
    manifest = parts.load_yaml("configs/split_v1.yaml")
    train_ids = [v["variant_id"] for v in manifest["variants"] if v["split"] == "train"]
    val_ids = [v["variant_id"] for v in manifest["variants"] if v["split"] == "val"]
    reference, reference_path = validation_reference(registry, outputs)
    if set(reference) != set(val_ids):
        raise ValueError("reference must cover exactly the validation variants")
    state, hashes = evaluation.git_state(), evaluation.file_hashes(CONFIG_FILES)
    code = evaluation.code_hash()
    tasks = [{"features": d, "seed": s, "train": train_ids, "val": val_ids, "reference": reference,
              "run_id": evaluation.new_run_id(f"p002fc-sarsa{d}-val-s{s}-{ENV_VERSION}")}
             for d in (16, 11) for s in TUNING_SEEDS]
    rows = []
    # Write each finished run immediately, so completed exploratory work survives interruption.
    with ProcessPoolExecutor(workers) as pool:
        for result in pool.map(run_one, tasks):
            t = result["task"]
            params = {"features": t["features"], "seed": t["seed"], "budget_steps": 200000,
                      "eval_every": 10000, "tail_fraction": .1, "alpha": .03 / t["features"],
                      "lambda": .9, "gamma": 1., "reward_scale": .01, "phi": "linear", "phi_alpha_s": 50.,
                      "epsilon": "linear .2 -> .01 over first half", "decision": "D-046"}
            extra = {"feature_check": True, "n_candidates": 48, "feature_count": t["features"],
                     "selection_normalized_cycle_time": result["score"], "history": result["history"],
                     "tail_count": result["tail_count"], "diverged": result["diverged"],
                     "wall_s": result["wall_s"], "steps_per_s": 200000 / result["wall_s"],
                     "secondary_final_weights": evaluation.aggregate(result["secondary"])}
            summary = result["summary"]
            records = [r | {"run_id": t["run_id"], "method": f"sarsa_features{t['features']}",
                            "seed": t["seed"], "split": "val"} for r in result["episodes"]]
            provenance = state | {"code_sha256": code, "config_files": hashes, "env_version": ENV_VERSION,
                                  "split_version": "v1", "reference": evaluation.file_hashes([reference_path]),
                                  "train_variants": train_ids, "val_variants": val_ids}
            out = Path(outputs) / t["run_id"]
            evaluation.write_run(out, params, provenance, records, summary)
            (out / "training.json").write_text(json.dumps({k: result[k] for k in
                ("history", "policy_weights", "final_weights", "tail_count", "episode_seeds", "secondary")}, indent=1), encoding="utf-8")
            row = evaluation.registry_row(t["run_id"], f"sarsa_features{t['features']}",
                evaluation.config_ref(hashes, params), "val", t["seed"], summary, extra, ENV_VERSION, "v1",
                evaluation.commit_label(state), status="exploratory", plan="002", budget_steps=200000)
            evaluation.append_registry(Path(registry), [row])
            rows.append(row)
            print(f"features={t['features']} seed={t['seed']} selection={result['score']:.6f} "
                  f"success={summary['success_rate']:.3f} diverged={result['diverged']} "
                  f"wall={result['wall_s']:.1f}s", flush=True)


def benchmark(repeats: int = 10000) -> None:
    """Three fixed train parts, full matrix calls, all available passes at reset; initialization excluded."""
    for fid in (46262, 65149, 124336):
        env = NBVEnv([f"{fid}_v0"])
        obs, _ = env.reset()
        feature, mask = ActionFeatures(env.nominal, env.k_max), env.action_masks()
        for _ in range(100):
            feature(obs, mask)
        started = time.perf_counter()
        for _ in range(repeats):
            feature(obs, mask)
        rate = repeats / (time.perf_counter() - started)
        print(f"{fid}: {rate:.1f} full feature matrices/s ({repeats} repeats)", flush=True)
