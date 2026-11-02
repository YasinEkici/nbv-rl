"""Phase 2 speed trial (Plan 001 Slice 6): throughput and resource use only, no learning results.

    uv run python scripts/phase2_speed_trial.py --policy mlp --n-envs 8 --steps 50000
    uv run python scripts/phase2_speed_trial.py --policy cnn3d --n-envs 8 --steps 20000

MaskablePPO on the NBV environment (48 candidates, train variants). mlp runs on the CPU (SB3 advises
CPU for PPO without a CNN); cnn3d uses a minimal 32^3 occupancy grid of the measured nominal points
on CUDA (the real voxel design belongs to Plan 003). Reports steps/s, CPU and GPU utilization, VRAM.
"""

import os

# Before numpy is imported: OpenBLAS commits per-thread buffers for all 24 cores in every process
# (~1.8 GB commit per env worker); single-threaded BLAS brings a worker down to ~0.36 GB.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import psutil  # noqa: E402

from nbv import features, parts  # noqa: E402
from nbv.env import NBVEnv  # noqa: E402

# torch and SB3 are imported inside main(): on Windows every SubprocVecEnv worker re-imports this
# module, and each process that loads CUDA torch commits a lot of memory (16 workers ran out).

GRID = 32


class VoxelObs(gym.ObservationWrapper):
    """Replace the measured-sample bits by a 32^3 occupancy grid of their nominal positions."""

    def __init__(self, env: NBVEnv):
        super().__init__(env)
        P = int(env.action_space.n)
        self.observation_space = gym.spaces.Dict({
            "voxels": gym.spaces.Box(0.0, 1.0, shape=(1, GRID, GRID, GRID), dtype=np.float32),
            "used": gym.spaces.MultiBinary(P),
            "q": env.observation_space["q"],
            "step": env.observation_space["step"],
        })

    def observation(self, obs):
        pts, _ = features.measured_points(obs, self.env.unwrapped.nominal)
        grid = np.zeros((1, GRID, GRID, GRID), np.float32)
        if len(pts):
            lo = np.array([-110.0, -110.0, -10.0])
            idx = np.clip(((pts - lo) / 220.0 * GRID).astype(int), 0, GRID - 1)
            grid[0, idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
        return {"voxels": grid, "used": obs["used"], "q": obs["q"], "step": obs["step"]}


def voxel_cnn_class():
    import torch
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    class Voxel3DCNN(BaseFeaturesExtractor):
        def __init__(self, space: gym.spaces.Dict, features_dim: int = 256):
            super().__init__(space, features_dim)
            self.cnn = torch.nn.Sequential(
                torch.nn.Conv3d(1, 16, 5, stride=2, padding=2), torch.nn.ReLU(),
                torch.nn.Conv3d(16, 32, 3, stride=2, padding=1), torch.nn.ReLU(),
                torch.nn.Conv3d(32, 64, 3, stride=2, padding=1), torch.nn.ReLU(),
                torch.nn.Flatten(),
            )
            rest = space["used"].n + 6 + 1
            self.head = torch.nn.Sequential(torch.nn.Linear(64 * 4 ** 3 + rest, features_dim), torch.nn.ReLU())

        def forward(self, obs):
            z = self.cnn(obs["voxels"])
            return self.head(torch.cat([z, obs["used"].float(), obs["q"], obs["step"] / 24.0], dim=1))

    return Voxel3DCNN


def make_env(variants, voxel: bool, rank: int):
    def thunk():
        e = NBVEnv(variants, n_candidates=48)
        e.reset(seed=rank)
        return VoxelObs(e) if voxel else e
    return thunk


class Monitor(threading.Thread):
    """Samples system CPU % and GPU util / memory (nvidia-smi) once per second."""

    def __init__(self):
        super().__init__(daemon=True)
        self.cpu, self.gpu, self.vram, self.ram, self.stop = [], [], [], [], False

    def run(self):
        psutil.cpu_percent(None)
        while not self.stop:
            time.sleep(1.0)
            self.cpu.append(psutil.cpu_percent(None))
            try:
                procs = [psutil.Process(), *psutil.Process().children(recursive=True)]
                self.ram.append(sum(p.memory_info().rss for p in procs) / 2**20)
            except psutil.Error:
                pass
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
                u, m = (float(x) for x in out.split(","))
                self.gpu.append(u)
                self.vram.append(m)
            except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
                pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", choices=["mlp", "cnn3d"], required=True)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--out", default="scratch/phase2_speed_trial.jsonl")
    args = ap.parse_args()
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(8)  # the env vars above would otherwise leave the learner single-threaded
    from stable_baselines3.common.vec_env import SubprocVecEnv

    train = [v["variant_id"] for v in parts.load_yaml("configs/split_v1.yaml")["variants"] if v["split"] == "train"]
    voxel = args.policy == "cnn3d"
    venv = SubprocVecEnv([make_env(train, voxel, i) for i in range(args.n_envs)])
    device = "cuda" if voxel else "cpu"
    kwargs = {"features_extractor_class": voxel_cnn_class()} if voxel else {}
    model = MaskablePPO("MultiInputPolicy", venv, n_steps=2048 // args.n_envs, batch_size=256, device=device,
                        policy_kwargs=kwargs, seed=0, verbose=0)
    mon = Monitor()
    mon.start()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model.learn(total_timesteps=args.steps)
    wall = time.perf_counter() - t0
    mon.stop = True
    venv.close()
    res = {
        "policy": args.policy, "device": device, "n_envs": args.n_envs, "steps": model.num_timesteps,
        "wall_s": round(wall, 1), "steps_per_s": round(model.num_timesteps / wall, 1),
        "cpu_percent_mean": round(float(np.mean(mon.cpu)), 1) if mon.cpu else None,
        "gpu_util_mean": round(float(np.mean(mon.gpu)), 1) if mon.gpu else None,
        "vram_used_mb_max": max(mon.vram) if mon.vram else None,
        "ram_rss_mb_max": round(max(mon.ram)) if mon.ram else None,
        "torch_max_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if device == "cuda" else None,
        "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
    }
    print(json.dumps(res))
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
