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
    uv run python scripts/precompute.py --config configs/<config>.yaml   # (planned, Plan 001)
    uv run python scripts/run.py --config configs/<config>.yaml          # (planned, Plan 001)

## Data

<Source of part meshes (ABC Dataset chunk / Thingi10K license-filtered subset), where to place them (data/raw/), and how the deviation generator creates variants.>
