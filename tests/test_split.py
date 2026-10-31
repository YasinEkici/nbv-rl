from collections import defaultdict

from nbv import parts

CONFIG = "configs/parts_v2.yaml"
MANIFEST = "configs/split_v1.yaml"


def test_variants_of_a_base_part_stay_in_one_split():
    m = parts.load_yaml(MANIFEST)
    part_split = {p["file_id"]: p["split"] for p in m["parts"]}
    seen = defaultdict(set)
    for v in m["variants"]:
        seen[v["base"]].add(v["split"])
    assert set(seen) == set(part_split)
    for fid, splits in seen.items():
        assert splits == {part_split[fid]}, fid


def test_counts_and_variants_per_part():
    m = parts.load_yaml(MANIFEST)
    assert m["counts"] == {"train": 8, "val": 3, "test": 3}
    per_base = defaultdict(int)
    for v in m["variants"]:
        per_base[v["base"]] += 1
    assert set(per_base.values()) == {m["variants_per_part"]}
    assert len({v["variant_id"] for v in m["variants"]}) == len(m["variants"])


def test_val_and_test_cover_every_stratum():
    m = parts.load_yaml(MANIFEST)
    for s in ("val", "test"):
        strata = sorted(p["stratum"] for p in m["parts"] if p["split"] == s)
        assert strata == [0, 1, 2], s


def test_manifest_matches_stratified_split():
    cfg = parts.load_yaml(CONFIG)
    sc = cfg["split"]
    hard = {p["file_id"]: p["hard_ratio"] for p in cfg["parts"]}
    split, stratum = parts.stratified_split(hard, sc["n_strata"], sc["n_val"], sc["n_test"], sc["seed"])
    m = parts.load_yaml(MANIFEST)
    assert {p["file_id"]: p["split"] for p in m["parts"]} == split
    assert {p["file_id"]: p["stratum"] for p in m["parts"]} == stratum


def test_share_alike_flag_matches_license():
    m = parts.load_yaml(MANIFEST)
    for p in m["parts"]:
        assert p["share_alike"] == (p["license"] == parts.SHARE_ALIKE_LICENSE)
