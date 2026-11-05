import importlib.util

import numpy as np


def load(name):
    spec = importlib.util.spec_from_file_location(name, f"scripts/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def records():
    rng = np.random.default_rng(0)
    out = []
    for method, base in (("ilp_tsp", 1.0), ("greedy_cost", 1.2)):
        for part in (10, 20, 30):
            for v in range(5):
                for seed in range(2):
                    ok = bool(rng.random() < 0.9)
                    out.append({"method": method, "part": part, "variant": f"{part}_v{v}", "seed": seed, "success": ok,
                                "normalized_cycle_time": base + 0.05 * rng.normal(), "cycle_time": 100 * base,
                                "passes_to_95": 10 if ok else None, "coverage_at_5": 0.6, "coverage_at_10": 0.9,
                                "auc_coverage_time": 0.5, "total_joint_motion": 30.0, "repair_passes": 0})
    return out


def test_report_tables_list_every_part_and_method():
    report = load("report")
    text = report.tables(records(), ["greedy_cost", "ilp_tsp"], n_boot=200)
    per_part, summary = text.split("### Split summary")
    assert "| method | 10 | 20 | 30 |" in per_part
    assert per_part.count("| greedy_cost |") == 1 and summary.count("| ilp_tsp |") == 1
    assert per_part.index("Per part") < text.index("Split summary")  # per-part table comes first


def test_bootstrap_coverage_runs():
    r = load("bootstrap_coverage").coverage(3, reps=10, n_boot=100, rng=np.random.default_rng(0))
    assert 0.0 <= r["coverage_of_true_mean"] <= 1.0 and r["median_width"] > 0


def test_feature_check_table_uses_registry_scores_and_predeclared_threshold():
    import json
    rows = []
    for seed in (1000, 1001, 1002):
        for d, value in ((16, 1.1), (11, 1.12)):
            rows.append({"status": "exploratory", "plan": "002", "env_version": "v4", "part_split": "val",
                         "seed": str(seed), "success_rate": "1.0", "extra_json": json.dumps({"feature_check": True,
                         "feature_count": d, "selection_normalized_cycle_time": value})})
    text = load("report").feature_check_table(rows)
    assert "1000" in text and "Mean" in text and "retain 16" in text and "0.020000" in text
