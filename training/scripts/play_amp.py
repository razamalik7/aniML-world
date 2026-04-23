"""Load an AMP-trained checkpoint and play it back in Isaac Sim.

Usage:
    .\\isaaclab.bat -p training\\scripts\\play_amp.py `
        --task AniML-Deer-USD-Walk-AMP-v0 `
        --num_envs 8 `
        --checkpoint C:\\path\\to\\model_N.pt

If --checkpoint is omitted, picks the latest model_*.pt under the newest
run directory in logs/rsl_rl/animl_deer_amp_walk.

Drops --headless so you can see the viewport. Use --headless for a
non-interactive pass if you just want to smoke-test that the checkpoint
loads without errors.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_path() -> None:
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _latest_checkpoint(experiment_dir: Path) -> Path:
    """Return the highest model_N.pt from the newest run directory."""
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"No experiment dir at {experiment_dir}")
    runs = sorted(
        (p for p in experiment_dir.iterdir() if p.is_dir()),
        key=lambda p: p.name,
        reverse=True,
    )
    if not runs:
        raise FileNotFoundError(f"No run dirs under {experiment_dir}")
    latest_run = runs[0]
    ckpt_re = re.compile(r"model_(\d+)\.pt$")
    ckpts = [p for p in latest_run.iterdir() if ckpt_re.match(p.name)]
    if not ckpts:
        raise FileNotFoundError(f"No model_N.pt under {latest_run}")
    ckpts.sort(key=lambda p: int(ckpt_re.match(p.name).group(1)))
    return ckpts[-1]


def main() -> None:
    _ensure_path()

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task", default="AniML-Deer-USD-Walk-AMP-v0")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--experiment", default="animl_deer_amp_walk",
        help="Used when --checkpoint is omitted. Look under "
             "logs/rsl_rl/<experiment>/ for the latest run.",
    )
    parser.add_argument(
        "--steps", type=int, default=2000,
        help="Number of sim steps to play before exiting.",
    )

    # Init AppLauncher before importing isaaclab (same as train_amp.py)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args, extras = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + extras
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Resolve checkpoint
    if args.checkpoint is None:
        exp_dir = _repo_root() / "logs" / "rsl_rl" / args.experiment
        ckpt = _latest_checkpoint(exp_dir)
        print(f"[play_amp] using latest ckpt: {ckpt}")
    else:
        ckpt = Path(args.checkpoint)
        if not ckpt.is_file():
            raise FileNotFoundError(f"No such checkpoint: {ckpt}")
        print(f"[play_amp] using ckpt: {ckpt}")

    # Imports that need AppLauncher alive
    import gymnasium as gym
    import torch

    import training.tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg

    from amp_rsl_rl.runners import AMPOnPolicyRunner

    # Load config dict from task registration
    import importlib
    task_spec = gym.spec(args.task)
    amp_cfg_ep = task_spec.kwargs.get("amp_rsl_rl_cfg_entry_point")
    if amp_cfg_ep is None:
        raise RuntimeError(f"Task {args.task!r} has no amp_rsl_rl_cfg_entry_point")
    module_name, attr_name = amp_cfg_ep.split(":")
    mod = importlib.import_module(module_name)
    runner_cfg = dict(getattr(mod, attr_name))

    # Build env. Disable command curriculum resetting during play so we see
    # a realistic distribution of commands the policy handles.
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    # Smaller episode length for play so we re-command periodically
    # env_cfg.episode_length_s = 10.0  # optional

    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    vec_env = RslRlVecEnvWrapper(env.unwrapped)  # type: ignore[arg-type]

    # Build runner and load
    runner = AMPOnPolicyRunner(
        env=vec_env,
        train_cfg=runner_cfg,
        log_dir=None,
        device=args.device,
    )
    runner.load(str(ckpt))
    policy = runner.get_inference_policy(device=args.device)

    # Play loop
    obs = vec_env.get_observations().to(args.device)
    print(f"[play_amp] starting playback for {args.steps} steps")
    for step in range(args.steps):
        with torch.inference_mode():
            actions = policy(obs)
        obs, _rewards, _dones, _ = vec_env.step(actions.to(vec_env.device))
        obs = obs.to(args.device)
        if step % 200 == 0:
            print(f"[play_amp] step {step}/{args.steps}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
