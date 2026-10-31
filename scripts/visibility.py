"""Dual-occlusion visibility on the nominal mesh and on every variant; fill the coverage columns.

    uv run python scripts/visibility.py

Needs the nominal robot plans from scripts/reachability.py. Both the nominal and the variant
visibility use the nominal passes (D-029): the plan comes from the CAD, the measurement from the part.
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
    ap.add_argument("--parts", default="configs/parts_v2.yaml")
    ap.add_argument("--manifest", default="configs/split_v1.yaml")
    ap.add_argument("--sensor", default="configs/sensor_v1.yaml")
    ap.add_argument("--candidates", default="configs/candidates_v1.yaml")
    ap.add_argument("--visibility", default="configs/visibility_v1.yaml")
    ap.add_argument("--robot-cache", default="data/cache/robot/nominal")
    ap.add_argument("--cache", default="data/cache/visibility")
    ap.add_argument("--report", default="results/reachability_v1.csv")
    args = ap.parse_args()

    pc, manifest = parts.load_yaml(args.parts), parts.load_yaml(args.manifest)
    sensor, cc, vc = (parts.load_yaml(p) for p in (args.sensor, args.candidates, args.visibility))
    g = visibility.SensorGeometry.from_config(sensor, vc)
    sizes = [cc["n_candidates"], cc["n_candidates_sensitivity"]]
    cache, robot_cache = Path(args.cache), Path(args.robot_cache)
    (cache / "nominal").mkdir(parents=True, exist_ok=True)
    geo = pc["geometry"]
    dense = cc["n_candidates_sensitivity"]
    values, timing = {}, []

    def run(mesh, p, nrm, plans, out_name):
        t0 = time.perf_counter()
        scene = visibility.make_scene(mesh, mesh.vertices[:, 2].min() - vc["table_offset_mm"])
        out = {}
        for n in sizes:
            vis = visibility.compute_visibility(scene, p, nrm, plans[n][0], g, vc["ray_offset_mm"])
            vis.save(cache / f"{out_name}_n{n}.npz")
            out[n] = vis
        timing.append({"mesh": out_name, "seconds": time.perf_counter() - t0})
        return out

    t_total = time.perf_counter()
    for fid in dict.fromkeys(v["base"] for v in manifest["variants"]):
        nominal = parts.load_base(Path(pc["paths"]["raw"]) / f"{fid}.ply", geo["longest_mm"], geo["max_edge_mm"])
        samples = visibility.sample_surface(nominal, vc["n_samples"], [vc["sample_seed"], fid])
        np.savez_compressed(cache / f"samples_{fid}.npz", face=samples.face, bary=samples.bary)
        plans = {}
        for n in sizes:
            z = np.load(robot_cache / f"{fid}_n{n}.npz")
            plans[n] = (ScanPasses.from_cache(z), z["reachable"])
        nom = run(nominal, *samples.on(nominal), plans, f"nominal/{fid}")
        nom_union = {n: visibility.union(nom[n].measured(), plans[n][1]) for n in sizes}
        for var in (v for v in manifest["variants"] if v["base"] == fid):
            vid = var["variant_id"]
            mesh = trimesh.load(Path(pc["paths"]["variants"]) / f"{vid}.ply", process=False)
            real = run(mesh, *samples.on(mesh), plans, vid)
            ref = visibility.union(real[dense].measured(), plans[dense][1])  # NSC reference (D-028)
            for n in sizes:
                rep = visibility.coverage_report(real[n], plans[n][1], ref)
                r_real = visibility.union(real[n].measured(), plans[n][1])
                rep["nominal_real_union_mismatch"] = (r_real ^ nom_union[n]).sum() / max(r_real.sum(), 1)
                values[(vid, str(n))] = rep
            print(vid, {n: round(values[(vid, str(n))]["observable_reachable_over_all"], 3) for n in sizes})
    t_total = time.perf_counter() - t_total
    update_csv(Path(args.report), values)
    (cache / "timing.json").write_text(json.dumps({"total_s": t_total, "per_mesh": timing}, indent=1), encoding="utf-8")
    print(f"total {t_total:.1f} s; wrote {args.report}")


if __name__ == "__main__":
    main()
