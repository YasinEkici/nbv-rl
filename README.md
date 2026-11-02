# NBV-RL: Learning Next-Best-View Scan Planning for Line-Laser Inspection

Course project (Deep Reinforcement Learning, graduate). A robot-mounted line-laser profilometer must scan cast parts with as few passes and as little arm motion as possible. We model scan planning as an MDP over candidate scan passes with dual-occlusion visibility and joint-space motion cost, train classical RL (Phase 1) and deep RL (Phase 2) agents on a family of part variants, and evaluate on unseen variants against greedy and ILP+TSP baselines.

## Status

| Phase | Description | Status |
|---|---|---|
| 0 | Repository scaffold | ✅ |
| 1 | Environment, precomputation, baselines (Plan 001) | 🔄 |
| 2 | Phase 1: classical RL (Plan 002) | ⏳ |
| 3 | Phase 2: deep RL + ablations (Plan 003) | ⏳ |
| 4 | Analysis and writing (Plan 004) | ⏳ |

Results: results/registry.csv (numbers are not duplicated here).

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.14 is pinned in `.python-version` (see docs/DECISIONS.md, D-009).

    uv sync

## Common commands

    uv run pytest -q
    uv run python scripts/prepare_parts.py --help
    uv run python scripts/reachability.py   # scan passes, IK mask, PTP times -> results/reachability_v1.csv
    uv run python scripts/visibility.py     # dual-occlusion visibility -> data/cache/visibility/, coverage columns
    uv run python scripts/env_benchmark.py  # environment speed and a greedy step-by-step breakdown
    uv run python scripts/d_plan.py         # planning difficulty per part -> results/d_plan_v1.csv
    uv run python scripts/run_baselines.py  # all baselines on train + val -> outputs/, results/registry.csv
    uv run python scripts/precompute.py --config configs/<config>.yaml   # (planned, Plan 001)
    uv run python scripts/run.py --config configs/<config>.yaml          # (planned, Plan 001)

## Data

Base parts are 14 mechanical meshes from [Thingi10K](https://huggingface.co/datasets/Thingi10K/Thingi10K) (Zhou & Jacobson, 2016), licensed CC BY or CC BY-SA, selected for occlusion features (internal cavities, ribs, deep holes, flanges). Each is scaled to a 200 mm longest side, and 5 cast-deviation variants per part are generated (low-frequency surface noise ≤ 0.5 mm, dents/bumps ≤ 1 mm, ±0.75% per-axis scale). Meshes and variants are not committed:

    uv run python scripts/prepare_parts.py fetch    --config configs/parts_v1.yaml   # -> data/raw/parts/
    uv run python scripts/prepare_parts.py split    --config configs/parts_v1.yaml   # -> configs/split_v1.yaml (committed)
    uv run python scripts/prepare_parts.py variants --config configs/parts_v2.yaml   # -> data/cache/variants_v2/

Attribution (from configs/split_v1.yaml). Variants derived from CC BY-SA parts must be shared under CC BY-SA if distributed.

| file_id | Part (Thingiverse thing) | Author | License | Split |
|---|---|---|---|---|
| 271875 | [PHIL](https://www.thingiverse.com/thing:136235) | Mr_MegaTronic | CC BY | val |
| 46262 | [PrintrBot Clone](https://www.thingiverse.com/thing:14130) | PacManFan | CC BY-SA | train |
| 124336 | [VEND - the totally printed candy dispenser](https://www.thingiverse.com/thing:45347) | Mr_MegaTronic | CC BY | train |
| 486859 | [Dremel Devil](https://www.thingiverse.com/thing:266035) | MAKE | CC BY | test |
| 178195 | [Turbine Rotary Tool 60000 rpm](https://www.thingiverse.com/thing:76369) | RichMac | CC BY | test |
| 548863 | [RAPIRO - The Humanoid Robot for your Raspberry Pi](https://www.thingiverse.com/thing:309466) | ShotaIshiwatari | CC BY | train |
| 37275 | [R. Maker - Mark I](https://www.thingiverse.com/thing:11654) | Mr_MegaTronic | CC BY-SA | val |
| 591247 | [3D printed mechanical Clock with Anchor Escapement](https://www.thingiverse.com/thing:328569) | TheGoofy | CC BY-SA | val |
| 65149 | [Functional Differential Gear System](https://www.thingiverse.com/thing:11836) | Thing-O-Fun | CC BY-SA | train |
| 702389 | [OpenRC 1:10 4WD Truggy Concept RC Car](https://www.thingiverse.com/thing:42198) | barspin | CC BY-SA | train |
| 135365 | [Customizable Battery Case (magnetic)](https://www.thingiverse.com/thing:51376) | walter | CC BY-SA | train |
| 124532 | [Colorimeter V0.0](https://www.thingiverse.com/thing:45443) | chowderhead | CC BY-SA | train |
| 635751 | [Adabot](https://www.thingiverse.com/thing:367226) | adafruit | CC BY-SA | train |
| 79729 | [Guitar Hero Whammy Bar Replacement Part](https://www.thingiverse.com/thing:26714) | falconphysics | CC BY-SA | test |
