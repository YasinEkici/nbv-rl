import numpy as np
import pytest

from nbv import evaluation


def steps(*rows):
    return [dict(zip(("t_move", "t_scan", "coverage"), r)) for r in rows]


def test_auc_step_function():
    st = steps((10.0, 10.0, 0.5), (20.0, 20.0, 1.0))
    # 0 for 20 s, 0.5 for 40 s, 1.0 for the remaining 120 s of the 180 s horizon.
    assert np.isclose(evaluation.auc_coverage_time(st), (0.5 * 40 + 120) / 180)
    assert evaluation.auc_coverage_time([]) == 0.0


def test_auc_clips_at_horizon():
    st = steps((100.0, 100.0, 1.0))
    assert evaluation.auc_coverage_time(st) == 0.0


def test_coverage_at_uses_final_value_after_end():
    st = steps((1, 1, 0.3), (1, 1, 0.6))
    assert evaluation.coverage_at(st, 1) == 0.3
    assert evaluation.coverage_at(st, 10) == 0.6


def test_hierarchical_bootstrap_constant_and_levels():
    const = {"a": {"v0": [2.0, 2.0], "v1": [2.0]}, "b": {"v0": [2.0]}}
    r = evaluation.hierarchical_bootstrap(const, n_boot=200)
    assert r["mean"] == r["ci_low"] == r["ci_high"] == 2.0
    data = {"a": {"v0": [0.0], "v1": [0.0]}, "b": {"v0": [10.0], "v1": [10.0]}}
    r = evaluation.hierarchical_bootstrap(data, n_boot=2000)
    assert r["mean"] == 5.0 and r["ci_low"] == 0.0 and r["ci_high"] == 10.0  # parts resampled


def test_nested_mean_weights_parts_equally():
    data = {"a": {f"v{i}": [1.0] for i in range(5)}, "b": {"v0": [3.0]}}
    assert evaluation.nested_mean(data) == 2.0


def test_paired_difference_matches_variants():
    a = {"p": {"v0": [3.0, 5.0], "v1": [1.0]}}
    b = {"p": {"v0": [1.0], "v2": [0.0]}}
    assert evaluation.paired_difference(a, b) == {"p": {"v0": [3.0]}}


def test_iqm():
    assert evaluation.iqm(np.arange(8)) == 3.5


def test_mark_superseded_keeps_rows(tmp_path):
    p = tmp_path / "registry.csv"
    evaluation.append_registry(p, [{"run_id": "a", "plan": "001", "status": "exploratory"},
                                   {"run_id": "b", "plan": "002", "status": "canonical"}])
    assert evaluation.mark_superseded(p, lambda r: r["plan"] == "001") == 1
    rows = list(__import__("csv").DictReader(open(p, encoding="utf-8")))
    assert [r["status"] for r in rows] == ["superseded", "canonical"]


def test_aggregate_normalized_cycle_time():
    ms = [{"success": True, "cycle_time": c, "passes_to_95": 1, "coverage_at_5": 0, "coverage_at_10": 0,
           "auc_coverage_time": 0, "nsc_at_5": 0, "nsc_at_10": 0, "raw_at_5": 0, "raw_at_10": 0,
           "total_joint_motion": 0, "repair_passes": 0, "normalized_cycle_time": r} for c, r in ((10, 1.0), (30, 1.5))]
    assert evaluation.aggregate(ms)["normalized_cycle_time"] == 1.25


def test_run_ids_are_unique_within_one_invocation():
    ids = {evaluation.new_run_id("p001x-random-n48-train-s0-v2", stamp="20261103T120000") for _ in range(2000)}
    assert len(ids) == 2000


def test_write_run_never_overwrites(tmp_path):
    out = tmp_path / "outputs" / "run-a"
    evaluation.write_run(out, {"a": 1}, {"commit": "x"}, [{"e": 1}], {"m": 1.0})
    before = (out / "episodes.jsonl").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        evaluation.write_run(out, {"a": 2}, {"commit": "y"}, [{"e": 2}], {"m": 2.0})
    assert (out / "episodes.jsonl").read_text(encoding="utf-8") == before


def test_registry_rejects_a_repeated_run_id(tmp_path):
    p = tmp_path / "registry.csv"
    evaluation.append_registry(p, [{"run_id": "a"}])
    with pytest.raises(ValueError):
        evaluation.append_registry(p, [{"run_id": "a"}])
    with pytest.raises(ValueError):
        evaluation.append_registry(tmp_path / "other.csv", [{"run_id": "b"}, {"run_id": "b"}])


def test_canonical_run_refused_on_dirty_tree():
    dirty = {"commit": "abc1234", "dirty": True, "changed_files": ["src/nbv/env.py"]}
    with pytest.raises(RuntimeError):
        evaluation.check_status("canonical", dirty)
    evaluation.check_status("exploratory", dirty)
    evaluation.check_status("canonical", dirty | {"dirty": False, "changed_files": []})
    assert evaluation.commit_label(dirty) == "abc1234-dirty"


def test_config_ref_names_files_and_hashes_contents(tmp_path):
    a = tmp_path / "a.yaml"
    a.write_text("x: 1\n", encoding="utf-8")
    h1 = evaluation.file_hashes([str(a)])
    ref = evaluation.config_ref(h1, {"n": 48})
    assert ref.startswith(f"{a};n=48 sha256:")
    a.write_text("x: 2\n", encoding="utf-8")
    assert evaluation.config_ref(evaluation.file_hashes([str(a)]), {"n": 48}) != ref
