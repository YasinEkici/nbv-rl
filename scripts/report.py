"""Result tables from the registry (D-039): per-part table first, then the split summary.

    uv run python scripts/report.py                                   # canonical rows, val, 48 candidates
    uv run python scripts/report.py --status exploratory --env-version v2 --split train

Reads the selected registry rows and their outputs/<run_id>/episodes.jsonl. Per part: the mean over
variants and seeds with a bootstrap interval over variants, then seeds. Split summary: mean of part
means with the hierarchical bootstrap interval (parts, variants, seeds); with few parts this
interval is narrower than its nominal level (scripts/bootstrap_coverage.py). Main table: the
canonical baseline set of D-041; appendix: the simple baselines; greedy-oracle only with --diagnostic.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from nbv import evaluation
from nbv.env import ENV_VERSION

MAIN = ["greedy_cost", "mcts", "ilp_tsp", "joint_opt"]
APPENDIX = ["random", "fixed", "greedy_ig"]
DIAGNOSTIC = ["greedy_oracle"]
SECONDARY = [("cycle_time", "cycle time (s)"), ("passes_to_95", "passes to 95%"), ("coverage_at_5", "coverage@5"),
             ("coverage_at_10", "coverage@10"), ("auc_coverage_time", "AUC"), ("total_joint_motion", "joint motion (rad)"),
             ("repair_passes", "repair passes")]


def load(rows: list[dict], outputs: Path) -> list[dict]:
    recs = []
    for r in rows:
        for line in open(outputs / r["run_id"] / "episodes.jsonl", encoding="utf-8"):
            e = json.loads(line)
            recs.append({k: e[k] for k in ("method", "part", "variant", "seed", "success", "normalized_cycle_time",
                                           *(s for s, _ in SECONDARY))})
    return recs


def fmt(r: dict, digits: int = 3) -> str:
    return f"{r['mean']:.{digits}f} [{r['ci_low']:.{digits}f}, {r['ci_high']:.{digits}f}]"


def stat(recs: list[dict], key: str, n_boot: int) -> dict | None:
    vals = [r for r in recs if r[key] is not None]
    if not vals:
        return None
    return evaluation.hierarchical_bootstrap(evaluation.nest(vals, key), n_boot=n_boot)


def tables(recs: list[dict], methods: list[str], n_boot: int) -> str:
    parts = sorted({r["part"] for r in recs}, key=str)
    out = ["### Per part: normalized cycle time (t / t_ILP+TSP), mean [interval over variants and seeds]; success rate", "",
           "| method | " + " | ".join(str(p) for p in parts) + " |", "|---|" + "---|" * len(parts)]
    for m in methods:
        cells = []
        for p in parts:
            rs = [r for r in recs if r["method"] == m and r["part"] == p]
            if not rs:
                cells.append("–")
                continue
            ncy = stat(rs, "normalized_cycle_time", n_boot)
            succ = evaluation.nested_mean(evaluation.nest(rs, "success"))
            cells.append(f"{fmt(ncy)}; {succ:.2f}")
        out.append(f"| {m} | " + " | ".join(cells) + " |")
    out += ["", "### Split summary: mean of part means [hierarchical bootstrap interval]", "",
            "| method | normalized cycle time | success rate | " + " | ".join(lbl for _, lbl in SECONDARY) + " |",
            "|---|---|---|" + "---|" * len(SECONDARY)]
    for m in methods:
        rs = [r for r in recs if r["method"] == m]
        if not rs:
            continue
        succ = evaluation.nested_mean(evaluation.nest(rs, "success"))
        sec = []
        for key, _ in SECONDARY:
            vals = [r for r in rs if r[key] is not None]
            sec.append(f"{evaluation.nested_mean(evaluation.nest(vals, key)):.3g}" if vals else "–")
        out.append(f"| {m} | {fmt(stat(rs, 'normalized_cycle_time', n_boot))} | {succ:.2f} | " + " | ".join(sec) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", default="results/registry.csv")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--status", default="canonical")
    ap.add_argument("--env-version", default=ENV_VERSION)
    ap.add_argument("--plan", default="001")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--diagnostic", action="store_true", help="add greedy-oracle")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--out", help="write the markdown here instead of stdout")
    args = ap.parse_args()
    rows = [r for r in csv.DictReader(open(args.registry, encoding="utf-8"))
            if r["status"] == args.status and r["env_version"] == args.env_version and r["plan"] == args.plan
            and r["part_split"] == args.split and json.loads(r["extra_json"]).get("n_candidates") == args.n]
    rows = list({r["run_id"]: r for r in rows}.values())  # pre-D-037 rows repeat run_ids
    if not rows:
        raise SystemExit(f"no registry rows with status={args.status}, env {args.env_version}, split {args.split}, n {args.n}")
    recs = load(rows, Path(args.outputs))
    methods = MAIN + APPENDIX + (DIAGNOSTIC if args.diagnostic else [])
    head = (f"## {args.split}, {args.n} candidates, {args.status}, ENV {args.env_version}: "
            f"{len(rows)} runs, {len(recs)} episodes, {len({r['part'] for r in recs})} parts\n\n"
            f"Main table: {', '.join(MAIN)}. Appendix: {', '.join(APPENDIX)}.\n\n")
    text = head + tables(recs, [m for m in methods if any(r["method"] == m for r in recs)], args.n_boot)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
