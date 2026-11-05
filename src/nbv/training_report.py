"""Registry-only smoke and tuning tables for Plan 002; no new evaluations."""

import csv
import json
from pathlib import Path

import numpy as np

from nbv.env import ENV_VERSION
from nbv.experiment import METHODS, StopRun, grid_settings, select_setting


def rows_for(tag: str, registry: str = "results/registry.csv") -> list[tuple[dict, dict]]:
    rows = list(csv.DictReader(open(registry, encoding="utf-8")))
    return [(r, json.loads(r["extra_json"])) for r in rows if r["plan"] == "002"
            and r["env_version"] == ENV_VERSION and r["status"] == "exploratory" and r["part_split"] == "val"
            and json.loads(r["extra_json"]).get("tag") == tag]


def official_hours(measurements: dict) -> float:
    groups = [(m, "linear", 11) for m in METHODS]
    groups += [("sarsa", p, 11) for p in ("none", "convex", "value_estimate")]
    groups += [("sarsa", "linear", 16)]
    return sum(10 * (measurements[g]["learning_s"] * 200 + measurements[g]["evaluation_s"])
               for g in groups) / 20 / 3600


def smoke_report(registry: str = "results/registry.csv", out: str = "results/training_speed_v1.md") -> float:
    rows = rows_for("p002smoke", registry)
    data = {}
    uncached = {}
    for row, extra in rows:
        key = row["method"], extra["phi"], extra["feature_count"]
        target = data if extra["cache_value"] else uncached
        if key in target:
            raise ValueError("duplicate smoke measurement; select an explicit invocation")
        target[key] = extra
    text = ["# Plan 002 single-core smoke measurements", "",
            "10000 training steps per row, 20 VAL evaluations. Learning steps/s excludes evaluation time;",
            "nominal model/feature initialization inside the learning loop is included. All rows are exploratory.", "",
            "| method | none | linear | convex | value_estimate (cached) | value_estimate (uncached) | cache speedup |",
            "|---|---|---|---|---|---|---|"]
    for method in METHODS:
        speeds = [data[method, p, 11]["steps_per_s"] for p in ("none", "linear", "convex", "value_estimate")]
        plain = uncached[method, "value_estimate", 11]["steps_per_s"]
        text.append(f"| {method} | " + " | ".join(f"{s:.1f}" for s in speeds)
                    + f" | {plain:.1f} | {speeds[-1]/plain:.3f}x |")
    estimate = official_hours(data)
    text += ["", f"16-feature linear SARSA: {data['sarsa', 'linear', 16]['steps_per_s']:.1f} learning steps/s.", "",
             f"Estimated official wall time on 20 cores: **{estimate:.3f} hours**.",
             "Includes 30 main seeds, 30 SARSA Phi-ablation seeds and 10 exploratory full-feature seeds (all 2000000 steps).",
             "Formula: sum over 70 seeds of (200 × smoke learning seconds + smoke evaluation seconds), divided by 20.",
             "This is an ideal parallel estimate; thermal throttling and contention can increase wall time."]
    Path(out).write_text("\n".join(text) + "\n", encoding="utf-8")
    print("\n".join(text), flush=True)
    if estimate > 12:
        raise StopRun("official-time estimate exceeds 12 hours")
    return estimate


def tuning_report(registry: str = "results/registry.csv", out: str = "results/tuning_v1.md",
                  settings_out: str = "results/selected_settings_v1.json") -> dict:
    rows = rows_for("p002t", registry)
    lookup = {}
    for row, extra in rows:
        key = extra["grid_id"], int(row["seed"])
        if key in lookup:
            raise ValueError("duplicate tuning run; select an explicit invocation")
        lookup[key] = extra
    text = ["# Plan 002 tuning grid (exploratory)", "",
            "VAL score = mean of the final two evaluations, using running tail-mean weights; lower is better.",
            "Spread is population standard deviation across seeds 1000–1002. No test-part evaluation.", ""]
    selected = {}
    for method in METHODS:
        text += [f"## {method}", "", "| point | setting | seed 1000 | seed 1001 | seed 1002 | mean | spread | diverged |",
                 "|---|---|---|---|---|---|---|---|"]
        points = []
        for i, setting in enumerate(grid_settings(method)):
            name = f"{method}-{i:02d}"
            extras = [lookup.get((name, seed)) for seed in (1000, 1001, 1002)]
            scores = [e["selection_normalized_cycle_time"] for e in extras if e is not None]
            values = [f"{e['selection_normalized_cycle_time']:.6f}" if e else "pending" for e in extras]
            diverged = sum(e["diverged"] for e in extras if e is not None)
            stats = f"{np.mean(scores):.6f} | {np.std(scores):.6f}" if len(scores) == 3 else "pending | pending"
            text.append(f"| {name} | {json.dumps(setting, sort_keys=True)} | " + " | ".join(values)
                        + f" | {stats} | {diverged}/{len(scores)} |")
            if len(scores) == 3:
                points.append({"grid_id": name, "setting": setting, "scores": scores, "diverged": diverged})
        if len(points) == 12:
            choice = select_setting(points)
            selected[method] = choice | {"mean": float(np.mean(choice["scores"])), "spread": float(np.std(choice["scores"]))}
            text += ["", f"Selected: {choice['grid_id']}, {json.dumps(choice['setting'], sort_keys=True)}.",
                     "Rule: within 0.005 of the minimum mean, lowest seed spread, then smallest alpha.", ""]
    text += ["## Selected settings and canonical VAL references", "",
             "| method | VAL selection score | seed spread |", "|---|---|---|"]
    for method, choice in selected.items():
        text.append(f"| {method} | {choice['mean']:.6f} | {choice['spread']:.6f} |")
    all_rows = list(csv.DictReader(open(registry, encoding="utf-8")))
    for method in ("greedy_cost", "mcts", "ilp_tsp"):
        refs = [json.loads(r["extra_json"])["normalized_cycle_time"] for r in all_rows
                if r["plan"] == "001" and r["status"] == "canonical" and r["env_version"] == ENV_VERSION
                and r["part_split"] == "val" and r["method"] == method
                and json.loads(r["extra_json"]).get("n_candidates") == 48]
        text.append(f"| {method} (reference) | {np.mean(refs):.3f} | — |")
    Path(out).write_text("\n".join(text) + "\n", encoding="utf-8")
    Path(settings_out).write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    print("\n".join(text), flush=True)
    return selected
