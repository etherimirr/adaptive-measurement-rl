"""Train one agent with one seed.

    python scripts/train.py --config configs/ppo.yaml --seed 0

Writes the model to ``results/models/<algo>_s<seed>.zip`` and per-episode
training logs (return, success, instrument time) to
``results/logs/<algo>_s<seed>.monitor.csv``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meas_rl.env import AdaptiveMeasureEnv  # noqa: E402

ALGOS = {"ppo": PPO, "sac": SAC}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=None, help="override config")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    algo = cfg["algo"]
    steps = args.timesteps or cfg["total_timesteps"]
    gamma = cfg["gamma"]

    # One thread per run: runs are launched side by side, and tiny MLPs gain
    # nothing from intra-op parallelism.
    torch.set_num_threads(1)
    set_random_seed(args.seed)

    log_dir = ROOT / "results" / "logs"
    model_dir = ROOT / "results" / "models"
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{cfg.get('name', algo)}_s{args.seed}"

    env = make_vec_env(
        AdaptiveMeasureEnv,
        n_envs=cfg["n_envs"],
        seed=args.seed,
        # Shaping uses the agent's own discount so it stays policy-invariant.
        env_kwargs={"shaping_gamma": gamma, **cfg.get("env_kwargs", {})},
        monitor_dir=str(log_dir / tag),
        monitor_kwargs={"info_keywords": ("success", "time_used", "n_measurements")},
    )

    model = ALGOS[algo](
        "MlpPolicy",
        env,
        gamma=gamma,
        seed=args.seed,
        verbose=0,
        device="cpu",
        **cfg["hyperparams"],
    )
    t0 = time.time()
    model.learn(total_timesteps=steps)
    elapsed = time.time() - t0
    model.save(model_dir / tag)

    meta = {
        "algo": algo,
        "seed": args.seed,
        "timesteps": steps,
        "train_seconds": round(elapsed, 1),
        "config": cfg,
    }
    (model_dir / f"{tag}.json").write_text(json.dumps(meta, indent=2))
    print(f"{tag}: {steps} steps in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
