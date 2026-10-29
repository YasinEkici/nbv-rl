"""Dual-occlusion visibility for every variant and candidate set; fill the coverage columns of the reports.

    uv run python scripts/visibility.py

Needs the robot-plan caches from scripts/reachability.py (pass geometry and reachability).
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import trimesh

from nbv import parts, visibility
from nbv.candidates import ScanPasses


def update_csv(path: Path, values: dict[tuple[str, str], dict]) -> None:
    """Fill columns of `path` for rows keyed by (variant_id, n_candidates)."""
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    fields = list(rows[0])
    for r in rows:
        for k, v in values.get((r["variant_id"], r["n_candidates"]), {}).items():
            if k not in fields:
                fields.append(k)
            r[k] = round(float(v), 4)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", default="configs/parts_v1.yaml")
    ap.add_argument("--manifest", default="configs/split_v1.yaml")
    ap.add_argument("--sensor", default="configs/sensor_v1.yaml")
    ap.add_argument("--candidates", default="configs/candidates_v1.yaml")
    ap.add_argument("--visibility", default="configs/visibility_v1.yaml")
    ap.add_argument("--robot-cache", default="data/cache/robot")
    ap.add_argument("--cache", default="data/cache/visibility")
    ap.add_argument("--report", default="results/reachability_v1.csv")
    ap.add_argument("--base-sweep", default="results/base_sweep")
    args = ap.parse_args()

    pc, manifest = parts.load_yaml(args.parts), parts.load_yaml(args.manifest)
    sensor, cc, vc = (parts.load_yaml(p) for p in (args.sensor, args.candidates, args.visibility))
    g = visibility.SensorGeometry.from_config(sensor, vc)
    sizes = [cc["n_candidates"], cc["n_candidates_sensitivity"]]
    cache, robot_cache = Path(args.cache), Path(args.robot_cache)
    cache.mkdir(parents=True, exist_ok=True)
    geo = pc["geometry"]

    samples, values, timing = {}, {}, []
    t_total = time.perf_counter()
    for var in manifest["variants"]:
        fid, vid = var["base"], var["variant_id"]
        if fid not in samples:
            base = parts.load_base(Path(pc["paths"]["raw"]) / f"{fid}.ply", geo["longest_mm"], geo["max_edge_mm"])
            samples[fid] = visibility.sample_surface(base, vc["n_samples"], [vc["sample_seed"], fid])
            np.savez_compressed(cache / f"samples_{fid}.npz", face=samples[fid].face, bary=samples[fid].bary)
        mesh = trimesh.load(Path(pc["paths"]["variants"]) / f"{vid}.ply", process=False)
        p, nrm = samples[fid].on(mesh)
        t0 = time.perf_counter()
        scene = visibility.make_scene(mesh, mesh.vertices[:, 2].min() - vc["table_offset_mm"])
        t_scene = time.perf_counter() - t0
        runs = {}
        for n in sizes:
            z = np.load(robot_cache / f"{vid}_n{n}.npz")
            t0 = time.perf_counter()
            vis = visibility.compute_visibility(scene, p, nrm, ScanPasses.from_cache(z), g, vc["ray_offset_mm"])
            dt = time.perf_counter() - t0
            vis.save(cache / f"{vid}_n{n}.npz")
            runs[n] = (vis, z["reachable"])
            timing.append({"variant_id": vid, "n": n, "scene_s": t_scene, "visibility_s": dt})
        dense = cc["n_candidates_sensitivity"]
        ref = visibility.union(runs[dense][0].measured(), runs[dense][1])  # NSC reference (D-028)
        for n, (vis, reachable) in runs.items():
            values[(vid, str(n))] = visibility.coverage_report(vis, reachable, ref)
        print(vid, {n: round(values[(vid, str(n))]["observable_reachable_over_all"], 3) for n in sizes})
    t_total = time.perf_counter() - t_total
    update_csv(Path(args.report), values)

    # Base-position sweep (train parts, 48 candidates): same visibility, different reachability.
    for csv_path in sorted(Path(args.base_sweep).glob("base_x*.csv")):
        bdir = robot_cache / csv_path.stem
        sweep_values = {}
        for r in csv.DictReader(open(csv_path, encoding="utf-8")):
            n = r["n_candidates"]
            vis = visibility.Visibility.load(cache / f"{r['variant_id']}_n{n}.npz")
            reachable = np.load(bdir / f"{r['variant_id']}_n{n}.npz")["reachable"]
            rep = visibility.coverage_report(vis, reachable)  # base sweep: primary ratio only
            sweep_values[(r["variant_id"], n)] = {"observable_reachable_over_all": rep["observable_reachable_over_all"]}
        update_csv(csv_path, sweep_values)

    (cache / "timing.json").write_text(json.dumps({"total_s": t_total, "per_run": timing}, indent=1), encoding="utf-8")
    print(f"total {t_total:.1f} s; wrote {args.report} and {args.base_sweep}/")


if __name__ == "__main__":
    main()
