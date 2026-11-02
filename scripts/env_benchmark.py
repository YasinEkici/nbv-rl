"""Environment speed (random policy) and a step-by-step greedy-oracle breakdown on one train variant.

    uv run python scripts/env_benchmark.py --variant 46262_v0

Greedy-oracle here is a diagnostic (it reads the privileged real visibility); the baseline proper is
implemented in Plan 001 Slice 5. "Nominal plan" = greedy on the nominal visibility until the nominal
coverage reaches the target, then evaluated on the real variant (censored if real coverage stays below it).
"""

import argparse
import json
import time

import numpy as np

from nbv import parts
from nbv.env import NBVEnv, evaluate_sequence


def greedy(env: NBVEnv, variant: str, use_real: bool) -> list[int]:
    obs, _ = env.reset(options={"variant": variant})
    vis = env.data.real.visibility if use_real else env.nominal.visibility
    ref = env.data.real.union if use_real else env.nominal.union
    covered, seq = np.zeros(vis.shape[1], bool), []
    while covered.sum() < env.target * ref.sum() - 1e-9:
        gain = (vis & ref & ~covered).sum(1).astype(float)
        mask = env.nominal.reachable.copy()
        mask[seq] = False
        gain[~mask] = -1
        if gain.max() <= 0:
            break
        a = int(np.argmax(gain))
        seq.append(a)
        covered |= vis[a]
    return seq


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="46262_v0")
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--out", default="scratch/env_benchmark.json")
    args = ap.parse_args()
    manifest = parts.load_yaml("configs/split_v1.yaml")
    train = [v["variant_id"] for v in manifest["variants"] if v["split"] == "train"]
    report = {}

    for n in (48, 128):
        env = NBVEnv(train, n_candidates=n)
        for vid in train:  # load caches before timing
            env.reset(options={"variant": vid})
        rng = np.random.default_rng(0)
        steps, t0 = 0, time.perf_counter()
        ep_times, ep_len, success = [], [], 0
        for ep in range(args.episodes):
            e0 = time.perf_counter()
            env.reset(seed=ep)
            done = False
            while not done:
                _, _, term, trunc, info = env.step(int(rng.choice(np.flatnonzero(env.action_masks()))))
                steps += 1
                done = term or trunc
            ep_times.append(time.perf_counter() - e0)
            ep_len.append(info["agent"]["cycle_time"])
            success += info["agent"]["success"]
        dt = time.perf_counter() - t0
        report[f"random_n{n}"] = {
            "steps_per_s": steps / dt, "episode_wall_ms_median": 1e3 * float(np.median(ep_times)),
            "steps_per_episode": steps / args.episodes, "success_rate": success / args.episodes,
            "cycle_time_s_median": float(np.median(ep_len)),
        }

    env = NBVEnv(train, n_candidates=48)
    oracle = evaluate_sequence(env, args.variant, greedy(env, args.variant, use_real=True))
    nominal_seq = greedy(env, args.variant, use_real=False)
    nominal_plan = evaluate_sequence(env, args.variant, nominal_seq)
    report["greedy_oracle"] = oracle
    report["nominal_plan"] = {"sequence": nominal_seq, **nominal_plan}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, default=float)
    print(json.dumps({k: v for k, v in report.items() if k.startswith("random")}, indent=1))
    for name in ("greedy_oracle", "nominal_plan"):
        r = report[name]
        print(f"\n{name}: success={r['success']} passes={r['passes_executed']} cycle_time={r['cycle_time']:.2f}s")
        print(" k pass  t_move  t_scan  t_home   cov    nsc    raw   nominal")
        for k, s in enumerate(r["steps"], 1):
            print(f"{k:2d} {s['pass']:4d} {s['t_move']:7.2f} {s['t_scan']:7.2f} {s['t_home']:7.2f} "
                  f"{s['coverage']:.3f}  {s['coverage_nsc']:.3f}  {s['coverage_raw']:.3f}  {s['nominal_coverage']:.3f}")


if __name__ == "__main__":
    main()
