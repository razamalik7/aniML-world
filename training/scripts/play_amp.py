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

# CRITICAL (ported from play_loco.py): h5py and Isaac Sim ship incompatible
# hdf5.dll files. Importing real h5py breaks Isaac Sim's generic_mo_io.dll;
# not stubbing it lets isaaclab_tasks' eager h5py import load the conflicting
# DLL. Install a stub so the import succeeds without the real DLL. Playback
# never uses HDF5 dataset I/O. Must run before any isaaclab import.
import sys
import types as _types
import importlib.machinery as _machinery
_h5py_stub = _types.ModuleType("h5py")
_h5py_stub.__spec__ = _machinery.ModuleSpec("h5py", loader=None)
_h5py_stub.__version__ = "3.16.0-stubbed"
class _H5pyAttrError:
    def __getattr__(self, name):
        raise AttributeError(f"h5py.{name} not available (stubbed for Isaac Sim DLL conflict)")
_h5py_stub.__getattr__ = _H5pyAttrError().__getattr__
_h5py_stub.File = _H5pyAttrError().__getattr__
_h5py_stub.Group = type("Group", (), {})
sys.modules["h5py"] = _h5py_stub
_errors_stub = _types.ModuleType("h5py._errors")
_errors_stub.__spec__ = _machinery.ModuleSpec("h5py._errors", loader=None)
sys.modules["h5py._errors"] = _errors_stub

import argparse
import os
import re
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
    parser.add_argument(
        "--realtime", action="store_true",
        help="Pace the playback loop to real-time (50 Hz, env_step_size=0.02s). "
             "Without this flag the loop runs as fast as the GPU can chew, "
             "which on an RTX 5080 is ~milliseconds total — the viewer window "
             "appears and disappears too quickly to watch. Use with --steps "
             "1500+ for ~30s of viewable playback.",
    )
    parser.add_argument(
        "--video", action="store_true",
        help="Record an MP4 of the playback via gymnasium.RecordVideo. "
             "Frames are captured offscreen via Isaac Sim's rgb_array render "
             "mode, so this works in --headless and avoids the unstable "
             "omni.hydra.rtx viewport on RTX 50-series + Isaac Sim 5.1.",
    )
    parser.add_argument(
        "--video_length", type=int, default=1500,
        help="Number of sim steps to record into the video (only used with "
             "--video). Default 1500 = 30s at 50 Hz.",
    )
    parser.add_argument(
        "--video_dir", type=str, default=None,
        help="Output directory for video. Defaults to <repo>/logs/videos/.",
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

    # render_mode="rgb_array" is required for video recording; otherwise None
    render_mode = "rgb_array" if args.video else None
    env = gym.make(args.task, cfg=env_cfg, render_mode=render_mode)
    vec_env = RslRlVecEnvWrapper(env.unwrapped)  # type: ignore[arg-type]

    # Set up trajectory dump if --video requested. We DON'T render in Isaac
    # Sim because the omni.hydra.rtx renderer is broken on RTX 50-series +
    # Isaac Sim 5.1 (crashes during init/shutdown of the Hydra engine, both
    # in viewer and headless+enable_cameras modes). Instead we dump joint
    # positions + root pose per step to a .npz, which a separate offline
    # tool (tools/render_playback_video.py) renders to MP4 via matplotlib
    # — completely bypassing Isaac Sim's rendering stack.
    traj_path = None
    traj_buffers = None
    if args.video:
        from datetime import datetime as _dt
        out_dir = (
            Path(args.video_dir) if args.video_dir
            else _repo_root() / "logs" / "videos"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = _dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        traj_path = out_dir / f"play_amp_{ts}.npz"
        traj_buffers = {
            "joint_positions": [],   # per-step list of (num_envs, 15)
            "root_pos": [],          # per-step list of (num_envs, 3)
            "root_quat_wxyz": [],    # per-step list of (num_envs, 4)
            "joints_list": None,     # populated on first step
        }
        # Override step count to match video length
        args.steps = max(args.steps, args.video_length + 10)
        print(f"[play_amp] will dump up to {args.video_length} steps of "
              f"trajectories to {traj_path}", flush=True)

    # Build runner and load
    runner = AMPOnPolicyRunner(
        env=vec_env,
        train_cfg=runner_cfg,
        log_dir=None,
        device=args.device,
    )
    runner.load(str(ckpt))
    policy = runner.get_inference_policy(device=args.device)

    # Pull a handle to the underlying Isaac Lab articulation so we can read
    # joint positions + root pose each step without going through any gym
    # wrapper chain.
    isaac_env = env.unwrapped
    robot_asset = isaac_env.scene["robot"]
    joint_names_in_scene = list(robot_asset.data.joint_names)
    if traj_buffers is not None:
        traj_buffers["joints_list"] = joint_names_in_scene

    # Play loop
    import time as _time
    obs = vec_env.get_observations().to(args.device)
    print(f"[play_amp] starting playback for {args.steps} steps "
          f"(realtime={args.realtime}, video={args.video})", flush=True)
    # Env step is 0.02s = 50 Hz. Sleep that much per loop to make the viewer
    # human-watchable. Without this, 1500 steps finish in ~1 sec on a 5080.
    target_dt = 0.02 if args.realtime else 0.0
    loop_start = _time.perf_counter()
    frames_dumped = 0
    for step in range(args.steps):
        with torch.inference_mode():
            actions = policy(obs)
        obs, _rewards, _dones, _ = vec_env.step(actions.to(vec_env.device))
        obs = obs.to(args.device)

        # Dump trajectory if recording
        if traj_buffers is not None and frames_dumped < args.video_length:
            jp = robot_asset.data.joint_pos.detach().cpu().numpy()       # (N, 15)
            rp = robot_asset.data.root_pos_w.detach().cpu().numpy()      # (N, 3)
            rq = robot_asset.data.root_quat_w.detach().cpu().numpy()     # (N, 4) wxyz
            traj_buffers["joint_positions"].append(jp.copy())
            traj_buffers["root_pos"].append(rp.copy())
            traj_buffers["root_quat_wxyz"].append(rq.copy())
            frames_dumped += 1

        if step % 200 == 0:
            print(f"[play_amp] step {step}/{args.steps} "
                  f"(dumped={frames_dumped})", flush=True)
        if target_dt > 0:
            # pace to wall clock so the viewer animates at realtime
            target = loop_start + (step + 1) * target_dt
            remaining = target - _time.perf_counter()
            if remaining > 0:
                _time.sleep(remaining)

    # Save trajectory dump
    if traj_buffers is not None and frames_dumped > 0:
        import numpy as _np
        _np.savez(
            str(traj_path),
            joint_positions=_np.stack(traj_buffers["joint_positions"]),  # (T, N, 15)
            root_pos=_np.stack(traj_buffers["root_pos"]),                # (T, N, 3)
            root_quat_wxyz=_np.stack(traj_buffers["root_quat_wxyz"]),    # (T, N, 4)
            joints_list=_np.array(traj_buffers["joints_list"]),
            fps=_np.array(50.0),
        )
        print(f"[play_amp] dumped {frames_dumped} frames to {traj_path}",
              flush=True)
        print(f"[play_amp] render with: python tools/render_playback_video.py "
              f"--input {traj_path}", flush=True)

    print(f"[play_amp] playback complete after {args.steps} steps", flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
