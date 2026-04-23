"""Launch amp-rsl-rl's AMPOnPolicyRunner on the deer walk task.

Unlike ``training/scripts/train.py``, this does NOT defer to Isaac Lab's
stock rsl_rl trainer -- it builds the env + runner directly so we can wire
``AMP_PPO`` and the ``amp`` observation group.

Usage (PowerShell, from Isaac Lab root):

    $env:PYTHONPATH = 'C:\\path\\to\\aniML_world'
    $env:ISAACLAB_ROOT = 'C:\\isaac-lab\\IsaacLab'
    .\\isaaclab.bat -p C:\\path\\to\\aniML_world\\training\\scripts\\train_amp.py `
        --task AniML-Deer-USD-Walk-AMP-v0 --num_envs 1024 --headless

Args:
    --task: Gym task ID (default: AniML-Deer-USD-Walk-AMP-v0)
    --num_envs: number of parallel envs (default: 1024)
    --max_iterations: override runner's max_iterations from config
    --headless: no GUI
    --seed: override runner seed
    --log_root: base directory for logs (default: logs/rsl_rl/<experiment_name>)

Smoke test recipe:
    --num_envs 64 --max_iterations 10 --headless

Note on dependencies:
    - amp-rsl-rl 1.2.0
    - rsl-rl-lib 3.0.0 (not 4.x/5.x -- amp-rsl-rl imports ``ActorCritic``
      which was removed after 3.x).
    - Isaac Lab's bundled rsl_rl may collide with this. If Isaac Lab vendors
      its own rsl_rl, set up a separate venv with the pinned versions and
      ensure it takes precedence.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_path() -> None:
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def main() -> None:
    _ensure_path()

    # Isaac Lab's AppLauncher parses its own CLI args, so we parse ours first
    # then pass the residue through.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task", default="AniML-Deer-USD-Walk-AMP-v0")
    parser.add_argument("--num_envs", type=int, default=2500)
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--task_reward_ratio",
        type=float,
        default=None,
        help=(
            "Override AMPOnPolicyRunner's hardcoded 50/50 blend of task vs "
            "style reward. If set to R, reward = R*task + (1-R)*style. Default "
            "None keeps the library's 0.5/0.5 blend. Useful values: 0.8 means "
            "task dominates, 0.2 means AMP style dominates."
        ),
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to a .pt checkpoint to warm-start from (calls runner.load()).",
    )
    parser.add_argument(
        "--log_root",
        default=str(_repo_root() / "logs" / "rsl_rl"),
        help="Root directory for logs. Default: <repo>/logs/rsl_rl.",
    )
    # NOTE: Do NOT add --device here -- AppLauncher.add_app_launcher_args
    # adds that. Use args.device (from AppLauncher) after parse.

    # --- Isaac Lab AppLauncher ---
    # Need to launch the simulator before importing isaaclab modules.
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args, extras = parser.parse_known_args()
    # Pass remaining args (e.g. --headless) to AppLauncher.
    sys.argv = [sys.argv[0]] + extras
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # --- Now safe to import Isaac Lab symbols ---
    import gymnasium as gym
    import torch

    import training.tasks  # noqa: F401 -- registers AniML-* gym tasks
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg

    from amp_rsl_rl.runners import AMPOnPolicyRunner

    # --- Resolve configs ---
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)

    # Load the AMP runner cfg dict from the registration entry point.
    task_spec = gym.spec(args.task)
    amp_cfg_ep = task_spec.kwargs.get("amp_rsl_rl_cfg_entry_point")
    if amp_cfg_ep is None:
        raise RuntimeError(
            f"Task {args.task!r} has no 'amp_rsl_rl_cfg_entry_point' in its "
            "gym registration. Use an AMP-enabled task."
        )
    module_name, attr_name = amp_cfg_ep.split(":")
    import importlib
    mod = importlib.import_module(module_name)
    runner_cfg: dict = dict(getattr(mod, attr_name))  # shallow copy so we can mutate

    if args.max_iterations is not None:
        runner_cfg["max_iterations"] = args.max_iterations
    if args.seed is not None:
        runner_cfg["seed"] = args.seed

    experiment_name = runner_cfg.get("experiment_name", "animl_deer_amp_walk")
    run_name = runner_cfg.get("run_name", "") or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path(args.log_root) / experiment_name / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train_amp] experiment={experiment_name}  run={run_name}")
    print(f"[train_amp] log_dir={log_dir}")
    print(f"[train_amp] num_envs={env_cfg.scene.num_envs}  device={args.device}")
    print(f"[train_amp] dataset path={runner_cfg['dataset']['amp_data_path']}")
    print(f"[train_amp] datasets={runner_cfg['dataset']['datasets']}")

    # --- Build env and runner ---
    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    # amp-rsl-rl expects a rsl_rl VecEnv. Isaac Lab provides the adapter.
    vec_env = RslRlVecEnvWrapper(env.unwrapped)  # type: ignore[arg-type]

    # Seed
    seed = runner_cfg.get("seed", 42)
    torch.manual_seed(seed)

    runner = AMPOnPolicyRunner(
        env=vec_env,
        train_cfg=runner_cfg,
        log_dir=str(log_dir),
        device=args.device,
    )

    # Monkey-patch the reward blend if the user overrode it. amp_on_policy_runner
    # hardcodes `rewards = 0.5 * rewards + 0.5 * style_rewards`; this patches it
    # to `R * rewards + (1-R) * style_rewards` by wrapping learn() internals.
    if args.task_reward_ratio is not None:
        ratio = float(args.task_reward_ratio)
        if not (0.0 <= ratio <= 1.0):
            raise ValueError(f"--task_reward_ratio must be in [0,1], got {ratio}")
        print(f"[train_amp] using task_reward_ratio={ratio} "
              f"(task={ratio:.2f}, style={1-ratio:.2f})")
        import amp_rsl_rl.runners.amp_on_policy_runner as _runner_mod
        # The simplest intervention: wrap discriminator.predict_reward so it
        # returns the STYLE reward scaled by (1-ratio)/0.5, and rely on the
        # existing 0.5/0.5 blend to then produce the correct combination.
        # reward = 0.5 * task + 0.5 * (style * (1-ratio)/0.5)
        #        = 0.5 * task + (1-ratio) * style
        # That's NOT ratio*task + (1-ratio)*style. Need to ALSO scale task.
        # Do that via env.step reward monkey patch — harder. Use a cleaner
        # approach: override runner.alg.process_env_step? No, rewards are
        # already blended before that.
        # Easiest: scale the style reward directly. Combined with the cfg
        # discriminator.reward_scale this gives us the effective blend.
        # Resulting blend: 0.5*task + 0.5*scale*D.  We want ratio*task +
        # (1-ratio)*D. So we set `task_scale_multiplier = 2*ratio` (applied
        # by overriding env.step wrapper) and `style_scale = 2*(1-ratio)`
        # (via patching predict_reward return).
        task_scale = 2.0 * ratio
        style_scale = 2.0 * (1.0 - ratio)

        orig_predict_reward = runner.discriminator.predict_reward

        def scaled_predict_reward(*a, **kw):
            return orig_predict_reward(*a, **kw) * style_scale
        runner.discriminator.predict_reward = scaled_predict_reward

        # For task: wrap vec_env.step so reward out *= task_scale.
        orig_step = vec_env.step

        def scaled_step(actions):
            obs, rewards, dones, extras = orig_step(actions)
            return obs, rewards * task_scale, dones, extras
        vec_env.step = scaled_step

    if args.resume_from is not None:
        resume_path = Path(args.resume_from)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume ckpt not found: {resume_path}")
        print(f"[train_amp] warm-starting from {resume_path}")
        runner.load(str(resume_path))

    runner.learn(num_learning_iterations=runner_cfg["max_iterations"])

    # Close cleanly
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
