"""Metrics, registry writing, and statistics (D-034).

Statistics are descriptive: the unit is the base part. With 3 test parts no significance test is
meaningful (the smallest two-sided paired Wilcoxon p-value at n = 3 is 0.25), and treating variants
or seeds as independent would be pseudo-replication. Results are reported per part, with a
hierarchical bootstrap CI (resample parts, then variants within a part, then seeds within a variant).
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

REGISTRY_FIELDS = [
    "run_id", "plan", "method", "config", "env_version", "split_version", "part_split", "seed", "budget_steps",
    "status", "success_rate", "cycle_time_to_95_s", "passes_to_95", "coverage_at_5", "coverage_at_10",
    "auc_coverage_time", "extra_json", "commit", "date",
]
AUC_HORIZON_S = 180.0  # PROTOCOL: [0, 180 s], step function, divided by 180


# --- Episode metrics -------------------------------------------------------------


def coverage_at(steps: list[dict], k: int, key: str = "coverage") -> float:
    """Coverage after k passes; the final value if the episode ended earlier (0 for no passes)."""
    if not steps:
        return 0.0
    return float(steps[min(k, len(steps)) - 1][key])


def auc_coverage_time(steps: list[dict], horizon_s: float = AUC_HORIZON_S) -> float:
    """Area under primary coverage vs time on [0, horizon], step function, divided by the horizon.

    Coverage jumps when a pass ends (cumulative move + scan time); the return home adds no coverage.
    """
    t, area, cov = 0.0, 0.0, 0.0
    for s in steps:
        t_end = t + s["t_move"] + s["t_scan"]
        area += cov * (min(t_end, horizon_s) - min(t, horizon_s))
        t, cov = t_end, s["coverage"]
    area += cov * max(horizon_s - t, 0.0)
    return area / horizon_s


def episode_metrics(res: dict) -> dict:
    """Per-episode metrics from an `nbv.env.evaluate_*` result (D-014, D-028)."""
    st = res["steps"]
    return {
        "success": res["success"], "cycle_time": res["cycle_time"], "passes_to_95": res["passes_to_target"],
        "coverage_at_5": coverage_at(st, 5), "coverage_at_10": coverage_at(st, 10),
        "nsc_at_5": coverage_at(st, 5, "coverage_nsc"), "nsc_at_10": coverage_at(st, 10, "coverage_nsc"),
        "raw_at_5": coverage_at(st, 5, "coverage_raw"), "raw_at_10": coverage_at(st, 10, "coverage_raw"),
        "auc_coverage_time": auc_coverage_time(st), "total_joint_motion": res["total_joint_motion"],
        "repair_passes": res["repair_passes"], "passes_executed": res["passes_executed"],
    }


def aggregate(metrics: list[dict]) -> dict:
    """Run-level summary over episodes (variants); cycle time includes censored episodes (D-014)."""
    ok = [m for m in metrics if m["success"]]
    mean = lambda k, ms=metrics: float(np.mean([m[k] for m in ms])) if ms else float("nan")  # noqa: E731
    return {
        "n_episodes": len(metrics), "success_rate": mean("success"), "cycle_time_to_95_s": mean("cycle_time"),
        "cycle_time_median_s": float(np.median([m["cycle_time"] for m in metrics])),
        "passes_to_95": mean("passes_to_95", ok), "coverage_at_5": mean("coverage_at_5"),
        "coverage_at_10": mean("coverage_at_10"), "auc_coverage_time": mean("auc_coverage_time"),
        "nsc_at_5": mean("nsc_at_5"), "nsc_at_10": mean("nsc_at_10"), "raw_at_5": mean("raw_at_5"),
        "raw_at_10": mean("raw_at_10"), "total_joint_motion_rad": mean("total_joint_motion"),
        "repair_passes": mean("repair_passes"),
    }


# --- Registry and run outputs ---------------------------------------------------------


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def config_hash(obj: dict) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def write_run(out_dir: Path, config: dict, provenance: dict, episodes: list[dict], summary: dict) -> None:
    """outputs/<run_id>/: config.yaml (as JSON-compatible YAML), metrics.json, provenance.json, episodes.jsonl."""
    import yaml

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=1), encoding="utf-8")
    with open(out_dir / "episodes.jsonl", "w", encoding="utf-8") as f:
        for e in episodes:
            f.write(json.dumps(e, default=float) + "\n")


def append_registry(path: Path, rows: list[dict]) -> None:
    """Append rows; existing rows are never rewritten (old ones are marked superseded by hand)."""
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in REGISTRY_FIELDS})


def registry_row(run_id: str, method: str, config: str, split: str, seed: int | str, summary: dict,
                 extra: dict, env_version: str, split_version: str, status: str = "exploratory",
                 plan: str = "001") -> dict:
    keys = ("success_rate", "cycle_time_to_95_s", "passes_to_95", "coverage_at_5", "coverage_at_10", "auc_coverage_time")
    return {
        "run_id": run_id, "plan": plan, "method": method, "config": config, "env_version": env_version,
        "split_version": split_version, "part_split": split, "seed": seed, "budget_steps": "", "status": status,
        **{k: "" if np.isnan(summary[k]) else round(summary[k], 4) for k in keys},  # undefined (no success) -> empty
        "extra_json": json.dumps(extra, sort_keys=True, default=float), "commit": git_commit(), "date": date.today().isoformat(),
    }


# --- Statistics (D-034) ------------------------------------------------------------------

Nested = dict[str, dict[str, list[float]]]  # part -> variant -> values over seeds


def nest(records: list[dict], value: str) -> Nested:
    """Group records with keys part, variant, value into part -> variant -> [values over seeds]."""
    out: Nested = defaultdict(lambda: defaultdict(list))
    for r in records:
        out[str(r["part"])][str(r["variant"])].append(float(r[value]))
    return {p: dict(v) for p, v in out.items()}


def nested_mean(data: Nested) -> float:
    """Mean of part means of variant means (each level weighted equally)."""
    return float(np.mean([np.mean([np.mean(s) for s in v.values()]) for v in data.values()]))


def per_part(data: Nested) -> dict[str, float]:
    return {p: float(np.mean([np.mean(s) for s in v.values()])) for p, v in data.items()}


def hierarchical_bootstrap(data: Nested, n_boot: int = 10_000, ci: float = 0.95, seed: int = 0) -> dict:
    """Mean with a hierarchical bootstrap CI: resample parts, then variants within each, then seeds.

    Follows Saravanan, Berman & Sober 2020 for nested data. With few parts the CI is wide by design.
    """
    rng = np.random.default_rng(seed)
    parts = [[np.asarray(s, float) for s in v.values()] for v in data.values()]
    stats = np.empty(n_boot)
    for b in range(n_boot):
        part_means = []
        for i in rng.integers(len(parts), size=len(parts)):
            variants = parts[i]
            var_means = [variants[j][rng.integers(len(variants[j]), size=len(variants[j]))].mean()
                         for j in rng.integers(len(variants), size=len(variants))]
            part_means.append(np.mean(var_means))
        stats[b] = np.mean(part_means)
    lo, hi = np.quantile(stats, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {"mean": nested_mean(data), "ci_low": float(lo), "ci_high": float(hi), "n_parts": len(parts)}


def paired_difference(a: Nested, b: Nested) -> Nested:
    """a − b on matching (part, variant); seeds are averaged per method first (seeds are not paired)."""
    return {p: {v: [float(np.mean(a[p][v]) - np.mean(b[p][v]))] for v in a[p] if v in b.get(p, {})} for p in a if p in b}


def iqm(values: np.ndarray) -> float:
    """Interquartile mean (mean of the middle 50%), kept for per-part summaries over many episodes."""
    x = np.sort(np.asarray(values, float))
    n = len(x)
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return float(x[lo:hi].mean()) if hi > lo else float(x.mean())
