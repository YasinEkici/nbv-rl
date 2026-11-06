"""Result tables from the registry (D-039): per-part table first, then the split summary.

    uv run python scripts/report.py --split train val --out results/baselines_v1.md   # canonical rows, 48 candidates
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


def feature_check_table(registry: list[dict]) -> str:
    """D-046 exploratory selection table; every displayed result comes from registry fields."""
    selected = {}
    for row in registry:
        extra = json.loads(row["extra_json"])
        if (row["status"] == "exploratory" and row["plan"] == "002" and row["env_version"] == ENV_VERSION
                and row["part_split"] == "val" and extra.get("feature_check")):
            key = (int(row["seed"]), extra["feature_count"])
            if key in selected:
                raise ValueError("multiple feature-check rows per setting/seed; select one predeclared invocation")
            selected[key] = (extra["selection_normalized_cycle_time"], float(row["success_rate"]))
    seeds = (1000, 1001, 1002)
    if set(selected) != {(s, d) for s in seeds for d in (16, 11)}:
        raise ValueError("expected the six D-046 feature-check registry rows")
    out = ["## Plan 002 exploratory feature check (D-046)", "",
           "VAL normalized cycle time; mean of evaluations at 190000 and 200000 steps using the running tail mean.", "",
           "| seed | 16 features | 11 features | improvement (11 minus 16) |", "|---|---|---|---|"]
    for seed in seeds:
        a, b = selected[seed, 16][0], selected[seed, 11][0]
        out.append(f"| {seed} | {a:.6f} | {b:.6f} | {b-a:.6f} |")
    a, b = (float(np.mean([selected[s, d][0] for s in seeds])) for d in (16, 11))
    out += [f"| Mean | {a:.6f} | {b:.6f} | {b-a:.6f} |", "",
            f"D-046 rule (improvement >= 0.01): {'retain 16' if b-a >= .01 else 'select 11; drop features 9 and 13–16'}.", ""]
    for d in (16, 11):
        success = np.mean([selected[s, d][1] for s in seeds])
        out.append(f"Final tail-policy success rate, {d} features: {success:.4f}.")
    for method in ("greedy_cost", "mcts"):
        values = [json.loads(r["extra_json"])["normalized_cycle_time"] for r in registry
                  if r["plan"] == "001" and r["status"] == "canonical" and r["env_version"] == ENV_VERSION
                  and r["part_split"] == "val" and r["method"] == method
                  and json.loads(r["extra_json"]).get("n_candidates") == 48]
        if values:
            out.append(f"Canonical val reference, {method}: {np.mean(values):.3f}.")
    out += ["", "These exploratory results select features only; they are not official method comparisons."]
    return "\n".join(out)


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
    ap.add_argument("--split", nargs="+", default=["val"], choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--diagnostic", action="store_true", help="add greedy-oracle")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--out", help="write the markdown here instead of stdout")
    ap.add_argument("--feature-check", action="store_true", help="D-046 exploratory table from registry scores")
    ap.add_argument("--training-speed", action="store_true", help="Slice 3 smoke table and official-time gate")
    ap.add_argument("--tuning", action="store_true", help="Slice 4 full grid and selected settings")
    ap.add_argument("--official-val", action="store_true", help="Slice 5a VAL table and exported learning curves")
    ap.add_argument('--pretest',action='store_true',help='Pretest curve diagnosis and recorded predictions')
    ap.add_argument('--frozen-train',action='store_true',help='Frozen canonical train evaluation table')
    ap.add_argument('--phase1-test',action='store_true',help='Single-opening test report and hypotheses')
    args = ap.parse_args()
    if args.pretest or args.frozen_train or args.phase1_test:
        from nbv.phase1_report import pretest_report, train_report, test_report
        if args.pretest: pretest_report(args.registry)
        elif args.frozen_train: train_report(args.registry)
        else: test_report(args.registry,n_boot=10000)
        return
    if args.official_val:
        from nbv.official_report import report
        report(args.registry)
        return
    if args.training_speed or args.tuning:
        from nbv.training_report import smoke_report, tuning_report
        if args.training_speed:
            smoke_report(args.registry, args.out or "results/training_speed_v1.md")
        else:
            tuning_report(args.registry, args.out or "results/tuning_v1.md")
        return
    registry = list(csv.DictReader(open(args.registry, encoding="utf-8")))
    if args.feature_check:
        text = feature_check_table(registry)
        if args.out:
            Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(text)
        return
    methods = MAIN + APPENDIX + (DIAGNOSTIC if args.diagnostic else [])
    sections = []
    for split in args.split:
        rows = [r for r in registry
                if r["status"] == args.status and r["env_version"] == args.env_version and r["plan"] == args.plan
                and r["part_split"] == split and json.loads(r["extra_json"]).get("n_candidates") == args.n]
        rows = list({r["run_id"]: r for r in rows}.values())  # pre-D-037 rows repeat run_ids
        if not rows:
            raise SystemExit(f"no registry rows with status={args.status}, env {args.env_version}, split {split}, n {args.n}")
        recs = load(rows, Path(args.outputs))
        head = (f"## {split}, {args.n} candidates, {args.status}, ENV {args.env_version}: "
                f"{len(rows)} runs, {len(recs)} episodes, {len({r['part'] for r in recs})} parts\n\n"
                f"Main table: {', '.join(MAIN)}. Appendix: {', '.join(APPENDIX)}.\n\n")
        sections.append(head + tables(recs, [m for m in methods if any(r["method"] == m for r in recs)], args.n_boot))
    text = "\n\n".join(sections)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
