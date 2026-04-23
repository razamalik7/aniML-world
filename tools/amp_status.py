"""Print a concise status line for the most recent AMP training run.

Usage:
    python tools/amp_status.py
    python tools/amp_status.py --experiment animl_deer_amp_walk_v2
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator


def latest_run(experiment: str) -> Path:
    base = Path(__file__).resolve().parent.parent / "logs" / "rsl_rl" / experiment
    runs = sorted(base.glob("*"), reverse=True)
    if not runs:
        raise FileNotFoundError(f"no runs under {base}")
    return runs[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default=None,
                        help="Default: newest of animl_deer_amp_walk_v2 | animl_deer_amp_walk")
    args = parser.parse_args()

    exps = [args.experiment] if args.experiment else [
        "animl_deer_amp_walk_v2", "animl_deer_amp_walk",
    ]
    runs = []
    for e in exps:
        try:
            runs.append((e, latest_run(e)))
        except FileNotFoundError:
            pass
    if not runs:
        raise SystemExit("no AMP runs found")

    # Print status for each experiment's newest run.
    for exp, run in runs:
        print(f"\n=== {exp}  run: {run.name} ===")
        ea = event_accumulator.EventAccumulator(str(run), size_guidance={event_accumulator.SCALARS: 4000})
        try:
            ea.Reload()
        except Exception as e:
            print(f"  failed to read events: {e}")
            continue

        def val(tag):
            ev = ea.Scalars(tag) if tag in ea.Tags()["scalars"] else []
            return ev

        reward = val("Train/mean_reward")
        if not reward:
            print("  (no metrics yet)")
            continue

        def last_and_peak(ev, window=20):
            if not ev:
                return None, None
            peak = max(e.value for e in ev)
            last_window = [e.value for e in ev[-window:]]
            last_avg = sum(last_window) / len(last_window)
            return last_avg, peak

        metrics = [
            ("Train/mean_reward", "reward"),
            ("Train/mean_task_reward", "task_r/step"),
            ("Train/mean_style_reward", "style_r/step"),
            ("Train/mean_episode_length", "ep_len"),
            ("Episode_Reward/track_lin_vel_xy_exp", "lin_vel_R"),
            ("Metrics/base_velocity/error_vel_xy", "err_vel_xy"),
            ("Episode_Termination/pelvis_too_low", "pelvis_too_low"),
            ("Policy/mean_noise_std", "noise_std"),
            ("Loss/amp_loss", "amp_loss"),
        ]
        n_iter = len(reward)
        print(f"  iterations recorded: {n_iter}")
        for tag, label in metrics:
            ev = val(tag)
            if ev:
                last_avg, peak = last_and_peak(ev)
                print(f"  {label:<18s} last20_avg={last_avg:+.4f}   peak={peak:+.4f}")


if __name__ == "__main__":
    main()
