"""Score every trained agent and every baseline on the same held-out devices.

    python scripts/evaluate.py            # 500 devices, all methods
    python scripts/evaluate.py --n 100    # quicker

Held-out devices are drawn from seeds 100000 upward, a range training never
touches. Every method sees the same devices in the same order. Writes
``results/eval.json`` and three figures under ``results/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meas_rl import physics  # noqa: E402
from meas_rl.baselines import InfoGainStopPolicy, RandomPolicy, SweepPolicy  # noqa: E402
from meas_rl.env import AdaptiveMeasureEnv  # noqa: E402

EVAL_SEED0 = 100_000
LABELS = {
    "ppo": "PPO end to end, continuous",
    "sac": "SAC end to end, continuous",
    "ppo_discrete": "PPO end to end, categorical",
    "ppo_dwellstop": "PPO dwell + stop, info-gain placement",
}
# (name prefix, colour) for each learned-agent family.
RL_GROUPS = (
    ("ppo", "tab:orange"),
    ("sac", "tab:green"),
    ("ppo_discrete", "tab:purple"),
    ("ppo_dwellstop", "tab:red"),
)
RESULTS = ROOT / "results"


# ----------------------------------------------------------------- rollouts


def rollout(env: AdaptiveMeasureEnv, act, seed: int) -> dict:
    """Run one episode on the device drawn from ``seed``; return its record."""
    obs, _ = env.reset(seed=seed)
    trace = []
    while True:
        action = act(obs, env)
        n_before = env.n_steps
        obs, _, term, trunc, info = env.step(action)
        if env.n_steps > n_before:  # a measurement was taken, not a stop
            trace.append((env.last_freq, env.last_dwell))
        if term or trunc:
            break
    lw = physics.linewidth(info["f0_true"], info["q_true"])
    return {
        "success": info["success"],
        "time_used": info["time_used"],
        "n_measurements": info["n_measurements"],
        "f0_err_mhz": info["f0_err"] * 1e3,
        "q_rel_err": info["q_rel_err"],
        "forced_stop": info["forced_stop"],
        # Detuning in linewidths and dwell for every measurement taken, used
        # to see whether a policy learned to linger near the resonance.
        "detune_lw": [abs(f - info["f0_true"]) / lw for f, _ in trace],
        "dwell": [d for _, d in trace],
    }


def eval_baseline(args) -> tuple[str, list[dict]]:
    policy, n = args
    env = AdaptiveMeasureEnv()
    records = []
    for k in range(n):
        policy.reset()
        records.append(rollout(env, lambda o, e: policy.act(e), EVAL_SEED0 + k))
    return policy.name, records


def eval_model(args) -> tuple[str, list[dict]]:
    path, n = args
    import torch
    from stable_baselines3 import PPO, SAC

    torch.set_num_threads(1)
    algo = PPO if Path(path).name.startswith("ppo") else SAC
    model = algo.load(path, device="cpu")
    meta = json.loads(Path(path).with_suffix(".json").read_text())
    env = AdaptiveMeasureEnv(**meta["config"].get("env_kwargs", {}))
    records = [
        rollout(env, lambda o, e: model.predict(o, deterministic=True)[0], EVAL_SEED0 + k)
        for k in range(n)
    ]
    return Path(path).stem, records


# ------------------------------------------------------------------ summary


def summarise(records: list[dict]) -> dict:
    t = np.array([r["time_used"] for r in records])
    return {
        "success_rate": float(np.mean([r["success"] for r in records])),
        "time_median_ms": float(np.median(t)),
        "time_mean_ms": float(np.mean(t)),
        "time_p90_ms": float(np.percentile(t, 90)),
        "measurements_mean": float(np.mean([r["n_measurements"] for r in records])),
        "f0_err_median_mhz": float(np.median([r["f0_err_mhz"] for r in records])),
        "q_rel_err_median": float(np.median([r["q_rel_err"] for r in records])),
        "forced_stop_rate": float(np.mean([r["forced_stop"] for r in records])),
    }


# Stopping thresholds (f0 std in GHz, log Q std) for the Bayesian rule, from
# loose to tight. Sweeping them traces the rule's time/accuracy frontier.
THRESHOLDS = ((0.0007, 0.08), (0.0005, 0.06), (0.0004, 0.05), (0.0003, 0.04), (0.0002, 0.03))


def baseline_suite() -> list:
    out = []
    for n_pts in (20, 30, 40):
        for dwell in (0.5, 1.0):
            out.append(SweepPolicy(n_pts, dwell))
    out.append(RandomPolicy(30, 1.0))
    for dwell in (1.0, 3.0):
        for f0_sd, lq_sd in THRESHOLDS:
            out.append(InfoGainStopPolicy(dwell, f0_std_stop=f0_sd, logq_std_stop=lq_sd))
    out.append(InfoGainStopPolicy(0.3))
    return out


# -------------------------------------------------------------------- plots


def plot_all(summary: dict, raw: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    # 1. success rate against instrument time: median (left) and mean (right).
    # The reward charges mean time, and the two panels differ exactly where a
    # method has a long tail of devices it keeps measuring.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    groups = {"sweep": "tab:gray", "random": "tab:brown", "infogain": "tab:blue"}
    for ax, key, label in (
        (axes[0], "time_median_ms", "median"),
        (axes[1], "time_mean_ms", "mean"),
    ):
        for name, s in summary["methods"].items():
            kind = name.split("(")[0]
            if kind in groups:
                ax.scatter(s[key], s["success_rate"], c=groups[kind], s=30)
        for dwell, style in (("1", "-"), ("3", "--")):
            pts = sorted(
                (s[key], s["success_rate"])
                for n, s in summary["methods"].items()
                if n.startswith(f"infogain(dwell={dwell},")
            )
            if pts:
                ax.plot(*zip(*pts), style, c="tab:blue", lw=1, alpha=0.7,
                        label=f"Bayesian stop, dwell {dwell} ms, threshold sweep")
        for algo, color in RL_GROUPS:
            seeds = [s for n, s in summary["methods"].items() if n.startswith(algo + "_s")]
            if not seeds:
                continue
            xs = [s[key] for s in seeds]
            ys = [s["success_rate"] for s in seeds]
            ax.scatter(xs, ys, c=color, s=18, alpha=0.5)
            ax.scatter(np.mean(xs), np.mean(ys), c=color, s=140, marker="*",
                       label=f"{LABELS[algo]} (star = mean of 5 seeds)")
        for kind, color in groups.items():
            ax.scatter([], [], c=color, label=kind)
        ax.set_xlabel(f"{label} instrument time per device (ms)")
        ax.set_ylim(0.45, 1.01)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("success rate (f0 within 1 MHz and Q within 10%)")
    axes[1].legend(fontsize=7.5, loc="lower right")
    fig.suptitle(f"Accuracy vs. measurement time, {summary['n_devices']} held-out devices")
    fig.tight_layout()
    fig.savefig(RESULTS / "pareto.png", dpi=150)
    plt.close(fig)

    # 2. learning curves from the training monitors
    fig, ax = plt.subplots(figsize=(7, 4))
    for algo, color in RL_GROUPS:
        curves = []
        for run_dir in sorted((RESULTS / "logs").glob(f"{algo}_s*")):
            frames = [pd.read_csv(f, skiprows=1) for f in sorted(run_dir.glob("*.monitor.csv"))]
            if not frames:
                continue
            df = pd.concat(frames).sort_values("t")
            df["steps"] = df["l"].cumsum()
            grid = np.linspace(0, df["steps"].iloc[-1], 100)
            roll = df["success"].astype(float).rolling(500, min_periods=50).mean()
            curves.append(np.interp(grid, df["steps"], roll.bfill()))
        if curves:
            c = np.array(curves)
            ax.plot(grid, c.mean(0), color=color, label=f"{LABELS[algo]}, mean of {len(c)} seeds")
            ax.fill_between(grid, c.min(0), c.max(0), color=color, alpha=0.15)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("training success rate (rolling 500 episodes)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "learning_curves.png", dpi=150)
    plt.close(fig)

    # 3. where the best learned policy spends its dwell time
    best = summary.get("best_rl")
    if best:
        recs = raw[best]
        det = np.concatenate([r["detune_lw"] for r in recs])
        dw = np.concatenate([r["dwell"] for r in recs])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(np.clip(det, 1e-2, None), dw, s=4, alpha=0.25, c="tab:red")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("distance from resonance (linewidths)")
        ax.set_ylabel("dwell chosen (ms)")
        ax.set_title(f"Dwell vs. detuning, {best}")
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(RESULTS / "dwell_profile.png", dpi=150)
        plt.close(fig)


# --------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    models = sorted(str(p) for p in (RESULTS / "models").glob("*.zip"))
    jobs_b = [(p, args.n) for p in baseline_suite()]
    jobs_m = [(m, args.n) for m in models]

    raw: dict[str, list[dict]] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name, recs in pool.map(eval_model, jobs_m):
            raw[name] = recs
        for name, recs in pool.map(eval_baseline, jobs_b):
            raw[name] = recs

    methods = {name: summarise(recs) for name, recs in raw.items()}
    aggregate = {}
    for algo, _ in RL_GROUPS:
        seeds = [methods[n] for n in methods if n.startswith(algo + "_s")]
        if seeds:
            aggregate[algo] = {
                k: {"mean": float(np.mean([s[k] for s in seeds])), "std": float(np.std([s[k] for s in seeds]))}
                for k in seeds[0]
            }
    # The dwell profile is drawn for the main agent's best seed.
    main = [n for n in methods if n.startswith("ppo_dwellstop_s")]
    best_rl = max(main, key=lambda n: (methods[n]["success_rate"], -methods[n]["time_median_ms"])) if main else None

    summary = {"n_devices": args.n, "eval_seed0": EVAL_SEED0, "methods": methods, "aggregate": aggregate, "best_rl": best_rl}
    (RESULTS / "eval.json").write_text(json.dumps(summary, indent=2))
    plot_all(summary, raw)

    width = max(len(n) for n in methods)
    print(f"{'method':{width}s}  success  t_med  t_p90  n_meas  f0err(MHz)  Qerr")
    for name, s in sorted(methods.items(), key=lambda kv: -kv[1]["success_rate"]):
        print(
            f"{name:{width}s}  {s['success_rate']:7.3f}  {s['time_median_ms']:5.1f}  {s['time_p90_ms']:5.1f}"
            f"  {s['measurements_mean']:6.1f}  {s['f0_err_median_mhz']:10.3f}  {s['q_rel_err_median']:.3f}"
        )
    for algo, agg in aggregate.items():
        print(
            f"{algo.upper()} over seeds: success {agg['success_rate']['mean']:.3f} ± {agg['success_rate']['std']:.3f},"
            f" median time {agg['time_median_ms']['mean']:.1f} ± {agg['time_median_ms']['std']:.1f} ms"
        )


if __name__ == "__main__":
    main()
