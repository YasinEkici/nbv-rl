"""Part meshes: loading, cast-deviation generator, and train/validation/test split.

Base parts come from Thingi10K (configs/parts_v1.yaml). `fetch_parts` is the only
function that touches the `thingi10k` package. It imports it lazily because on Windows
`thingi10k` (via lagrange) fails to load if open3d was imported first in the same
process; everything else in this module works on PLY files with trimesh only.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import trimesh
import yaml

THING_URL = "https://www.thingiverse.com/thing:{}"
SHARE_ALIKE_LICENSE = "Creative Commons - Attribution - Share Alike"


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def write_yaml(data: dict, path: str | Path, header: str = "") -> None:
    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    Path(path).write_text(header + text, encoding="utf-8")


# --- Source ------------------------------------------------------------------


def fetch_parts(file_ids: list[int], out_dir: str | Path, cache_dir: str | Path) -> list[dict]:
    """Download base parts from Thingi10K, write `<file_id>.ply`, return their metadata."""
    import thingi10k  # lazy: must be imported before open3d (see module docstring)

    thingi10k.init(cache_dir=str(cache_dir))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = {int(e["file_id"]): e for e in thingi10k.dataset(file_id=list(file_ids))}
    meta = []
    for fid in file_ids:
        e = entries[fid]
        v, f = thingi10k.load_file(e["file_path"])
        mesh = trimesh.Trimesh(v, f, process=False)
        check_mesh(mesh, name=str(fid))
        mesh.export(out_dir / f"{fid}.ply")
        meta.append(
            {
                "file_id": fid,
                "thing_id": int(e["thing_id"]),
                "thing_url": THING_URL.format(e["thing_id"]),
                "name": e["name"].strip().replace("&#039;", "'"),
                "author": e["author"],
                "license": e["license"],
                "share_alike": e["license"] == SHARE_ALIKE_LICENSE,
                "original_extent": [round(float(x), 2) for x in mesh.extents],
                "num_faces": len(f),
            }
        )
    return meta


# --- Mesh utilities ----------------------------------------------------------


def check_mesh(mesh: trimesh.Trimesh, name: str = "mesh") -> None:
    """Raise ValueError unless the mesh is watertight, winding-consistent, and has volume > 0."""
    if not mesh.is_watertight:
        raise ValueError(f"{name}: not watertight")
    if not mesh.is_winding_consistent:
        raise ValueError(f"{name}: inconsistent winding")
    if not mesh.volume > 0:
        raise ValueError(f"{name}: non-positive volume")


def normalize(mesh: trimesh.Trimesh, longest_mm: float) -> trimesh.Trimesh:
    """Scale uniformly so the longest bbox side is `longest_mm`; bbox centered in x/y, min z = 0."""
    s = longest_mm / mesh.extents.max()
    v = mesh.vertices * s
    lo, hi = v.min(0), v.max(0)
    v = v - np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    return trimesh.Trimesh(v, mesh.faces.copy(), process=False)


def subdivide(mesh: trimesh.Trimesh, max_edge_mm: float) -> trimesh.Trimesh:
    """Split edges until none is longer than `max_edge_mm`, so displacements can follow the surface."""
    v, f = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=max_edge_mm, max_iter=20)
    return trimesh.Trimesh(v, f, process=False)


def load_base(path: str | Path, longest_mm: float, max_edge_mm: float) -> trimesh.Trimesh:
    """Load a base part, normalize its size, and subdivide it for the deviation generator."""
    mesh = trimesh.load(path, process=False)
    out = subdivide(normalize(mesh, longest_mm), max_edge_mm)
    check_mesh(out, name=str(path))
    return out


def mesh_hash(mesh: trimesh.Trimesh, decimals: int = 6) -> str:
    """SHA-256 of rounded vertices (float64) and faces (int64)."""
    v = np.ascontiguousarray(np.round(mesh.vertices, decimals) + 0.0, dtype="<f8")  # +0.0 folds -0.0
    f = np.ascontiguousarray(mesh.faces, dtype="<i8")
    return hashlib.sha256(v.tobytes() + f.tobytes()).hexdigest()


# --- Deviation generator -----------------------------------------------------


def variant_rng(seed: int, file_id: int, index: int) -> np.random.Generator:
    """Independent stream per (seed, part, variant), so variants do not depend on generation order."""
    return np.random.default_rng([seed, file_id, index])


def _fourier_noise(v: np.ndarray, p: dict, rng: np.random.Generator) -> np.ndarray:
    """Sum of random plane waves; |result| <= amplitude_mm everywhere."""
    k = p["n_modes"]
    d = rng.normal(size=(k, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    lam = rng.uniform(*p["wavelength_mm"], size=k)
    phase = rng.uniform(0, 2 * np.pi, size=k)
    w = rng.uniform(0.5, 1.0, size=k)
    field = np.sin(2 * np.pi * (v @ d.T) / lam + phase) @ w
    return p["amplitude_mm"] * field / w.sum()


def _dents(mesh: trimesh.Trimesh, p: dict, rng: np.random.Generator) -> np.ndarray:
    """Gaussian dents/bumps at area-sampled surface points; |result| <= amplitude_mm.

    Each dent is weighted by the agreement between vertex and dent normals, so the
    opposite side of a thin wall is not pushed in the other direction.
    """
    n = int(rng.integers(p["count"][0], p["count"][1] + 1))
    area = mesh.area_faces
    fi = rng.choice(len(area), size=n, p=area / area.sum())
    bary = rng.dirichlet(np.ones(3), size=n)
    centers = np.einsum("ij,ijk->ik", bary, mesh.triangles[fi])
    normals = mesh.face_normals[fi]
    sigma = rng.uniform(*p["radius_mm"], size=n)
    amp = rng.uniform(-p["amplitude_mm"], p["amplitude_mm"], size=n)
    v, vn = mesh.vertices, mesh.vertex_normals
    out = np.zeros(len(v))
    for c, nc, s, a in zip(centers, normals, sigma, amp):
        r2 = ((v - c) ** 2).sum(1)
        out += a * np.exp(-r2 / (2 * s * s)) * np.clip(vn @ nc, 0.0, 1.0)
    return np.clip(out, -p["amplitude_mm"], p["amplitude_mm"])


def make_variant(base: trimesh.Trimesh, params: dict, rng: np.random.Generator) -> trimesh.Trimesh:
    """Cast-deviation variant: low-frequency noise + dents/bumps along normals, then per-axis scaling.

    `base` must come from `load_base` (normalized, subdivided). Topology is unchanged.
    """
    disp = _fourier_noise(base.vertices, params["noise"], rng) + _dents(base, params["dents"], rng)
    v = base.vertices + disp[:, None] * base.vertex_normals
    a = params["axis_scale"]
    v = v * (1.0 + rng.uniform(-a, a, size=3))  # origin is the bbox bottom center
    out = trimesh.Trimesh(v, base.faces.copy(), process=False)
    check_mesh(out, name="variant")
    return out


# --- Split -------------------------------------------------------------------


def stratified_split(
    hard: dict[int, float], n_strata: int, n_val: int, n_test: int, seed: int
) -> tuple[dict[int, str], dict[int, int]]:
    """Assign base parts to train/val/test, stratified by hard-to-see ratio.

    Parts are sorted by (hard, file_id) and cut into `n_strata` near-equal strata
    (stratum 0 = easiest). Each stratum contributes n_val/n_strata parts to val and
    n_test/n_strata to test, drawn at random; the rest go to train.
    """
    if n_val % n_strata or n_test % n_strata:
        raise ValueError("n_val and n_test must be multiples of n_strata")
    ids = sorted(hard, key=lambda i: (hard[i], i))
    strata = np.array_split(np.array(ids), n_strata)
    rng = np.random.default_rng(seed)
    nv, nt = n_val // n_strata, n_test // n_strata
    split, stratum = {}, {}
    for s, members in enumerate(strata):
        if len(members) <= nv + nt:
            raise ValueError(f"stratum {s} too small")
        perm = [int(i) for i in rng.permutation(members)]
        for j, fid in enumerate(perm):
            split[fid] = "val" if j < nv else "test" if j < nv + nt else "train"
            stratum[fid] = s
    return split, stratum


def build_manifest(cfg: dict, meta: list[dict], config_path: str) -> dict:
    """Split manifest: per-part source/license/split plus the variant list."""
    sc = cfg["split"]
    hard = {p["file_id"]: p["hard_ratio"] for p in cfg["parts"]}
    split, stratum = stratified_split(hard, sc["n_strata"], sc["n_val"], sc["n_test"], sc["seed"])
    by_id = {m["file_id"]: m for m in meta}
    parts = []
    for p in cfg["parts"]:
        fid = p["file_id"]
        m = by_id[fid]
        parts.append(
            {
                "file_id": fid,
                "thing_id": m["thing_id"],
                "thing_url": m["thing_url"],
                "name": m["name"],
                "author": m["author"],
                "license": m["license"],
                "share_alike": m["share_alike"],
                "original_unit": p["original_unit"],
                "original_extent": m["original_extent"],
                "hard_ratio": p["hard_ratio"],
                "stratum": stratum[fid],
                "split": split[fid],
            }
        )
    n_var = cfg["variants"]["per_part"]
    variants = [
        {"variant_id": f"{p['file_id']}_v{k}", "base": p["file_id"], "index": k, "split": p["split"]}
        for p in parts
        for k in range(n_var)
    ]
    return {
        "split_version": cfg["split_version"],
        "config": config_path,
        "source": cfg["source"],
        "deviation_seed": cfg["variants"]["seed"],
        "variants_per_part": n_var,
        "counts": {s: sum(p["split"] == s for p in parts) for s in ("train", "val", "test")},
        "parts": parts,
        "variants": variants,
    }


def generate_variants(cfg: dict, manifest: dict, raw_dir: str | Path, out_dir: str | Path) -> dict[str, str]:
    """Write every manifest variant as `<variant_id>.ply`; return {variant_id: mesh_hash}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    g = cfg["geometry"]
    seed = manifest["deviation_seed"]
    hashes = {}
    bases = {}
    for var in manifest["variants"]:
        fid = var["base"]
        if fid not in bases:
            bases = {fid: load_base(Path(raw_dir) / f"{fid}.ply", g["longest_mm"], g["max_edge_mm"])}
        mesh = make_variant(bases[fid], cfg["deviation"], variant_rng(seed, fid, var["index"]))
        mesh.export(out_dir / f"{var['variant_id']}.ply")
        hashes[var["variant_id"]] = mesh_hash(mesh)
    return hashes
