"""Playback for vanilla-PPO locomotion stages (Loco-Stage1/2/3).

Adapted from play_amp.py but uses ``rsl_rl.runners.OnPolicyRunner`` instead
of the AMP runner, since these stages use stock PPO without a discriminator.

Dumps trajectories to .npz for the offline meshed renderer
(tools/render_playback_meshed.py) — bypasses Isaac Sim's broken viewer
on RTX 50-series.

Usage:
    isaaclab.bat -p training/scripts/play_loco.py \
        --task AniML-Deer-USD-Loco-Stage3-v0 \
        --num_envs 4 --video --video_length 1500 \
        --checkpoint <abs_path>/model_1999.pt
"""

from __future__ import annotations

# CRITICAL: h5py and Isaac Sim ship incompatible hdf5.dll files. Importing
# real h5py before AppLauncher fixes h5py's ABI issue but BREAKS Isaac
# Sim's generic_mo_io.dll which depends on Isaac Sim's hdf5.dll. They
# can't coexist. But isaaclab_tasks only imports h5py via the auto-loaded
# datasets module (utils/datasets/__init__.py → hdf5_dataset_file_handler).
# We don't use HDF5 dataset I/O for playback. Install a stub h5py in
# sys.modules so the eager import succeeds without loading the conflicting
# DLL. parse_cfg.py and friends just need the import to succeed; they
# never actually call h5py.File() during playback.
import sys
import types as _types
import importlib.machinery as _machinery
_h5py_stub = _types.ModuleType("h5py")
_h5py_stub.__spec__ = _machinery.ModuleSpec("h5py", loader=None)
_h5py_stub.__version__ = "3.16.0-stubbed"
class _H5pyAttrError:
    """Raise AttributeError on any access — surface the bug if anything
    actually tries to use h5py at runtime."""
    def __getattr__(self, name):
        raise AttributeError(
            f"h5py.{name} not available (stubbed for Isaac Sim DLL conflict; "
            "playback doesn't need HDF5 dataset I/O)"
        )
_h5py_stub.__getattr__ = _H5pyAttrError().__getattr__
_h5py_stub.File = _H5pyAttrError().__getattr__
_h5py_stub.Group = type("Group", (), {})  # used in isinstance() checks; safe sentinel
sys.modules["h5py"] = _h5py_stub
_errors_stub = _types.ModuleType("h5py._errors")
_errors_stub.__spec__ = _machinery.ModuleSpec("h5py._errors", loader=None)
sys.modules["h5py._errors"] = _errors_stub

import argparse
import re
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_path() -> None:
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def main() -> None:
    _ensure_path()

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task", default="AniML-Deer-USD-Loco-Stage1-v0")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video_length", type=int, default=1500)
    parser.add_argument("--video_dir", type=str, default=None)

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args, extras = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + extras
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        raise FileNotFoundError(f"No such checkpoint: {ckpt}")
    print(f"[play_loco] using ckpt: {ckpt}")

    import gymnasium as gym
    import torch

    import training.tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils import parse_env_cfg
    from rsl_rl.runners import OnPolicyRunner

    import importlib
    import importlib.metadata
    task_spec = gym.spec(args.task)
    rsl_cfg_ep = task_spec.kwargs.get("rsl_rl_cfg_entry_point")
    if rsl_cfg_ep is None:
        raise RuntimeError(f"Task {args.task!r} has no rsl_rl_cfg_entry_point")
    module_name, attr_name = rsl_cfg_ep.split(":")
    mod = importlib.import_module(module_name)
    runner_cfg_class = getattr(mod, attr_name)
    runner_cfg = runner_cfg_class()  # instantiate the dataclass

    # Apply Isaac Lab's deprecation shim so the to_dict() output matches what
    # rsl_rl 5.0+ OnPolicyRunner / PPO.construct_algorithm expect (it adds the
    # `class_name` fields under actor/critic etc).
    rsl_rl_version = importlib.metadata.version("rsl-rl-lib")
    runner_cfg = handle_deprecated_rsl_rl_cfg(runner_cfg, rsl_rl_version)

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)

    env = gym.make(args.task, cfg=env_cfg)
    vec_env = RslRlVecEnvWrapper(env.unwrapped)  # type: ignore[arg-type]

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
        traj_path = out_dir / f"play_loco_{ts}.npz"
        traj_buffers = {
            "joint_positions": [],
            "root_pos": [],
            "root_quat_wxyz": [],
            "joints_list": None,
        }
        args.steps = max(args.steps, args.video_length + 10)
        print(f"[play_loco] will dump up to {args.video_length} steps to {traj_path}",
              flush=True)

    # Build runner — vanilla rsl_rl 5.0+ OnPolicyRunner.
    runner = OnPolicyRunner(
        env=vec_env,
        train_cfg=runner_cfg.to_dict(),
        log_dir=None,
        device=args.device,
    )
    runner.load(str(ckpt))
    policy = runner.get_inference_policy(device=args.device)

    isaac_env = env.unwrapped
    robot_asset = isaac_env.scene["robot"]
    joint_names_in_scene = list(robot_asset.data.joint_names)
    if traj_buffers is not None:
        traj_buffers["joints_list"] = joint_names_in_scene
        print(f"[play_loco] joints recognized: {len(joint_names_in_scene)}", flush=True)

    obs = vec_env.get_observations().to(args.device)
    print(f"[play_loco] starting playback for {args.steps} steps", flush=True)
    frames_dumped = 0
    for step in range(args.steps):
        with torch.inference_mode():
            actions = policy(obs)
        obs, _rewards, _dones, _ = vec_env.step(actions.to(vec_env.device))
        obs = obs.to(args.device)

        if traj_buffers is not None and frames_dumped < args.video_length:
            jp = robot_asset.data.joint_pos.detach().cpu().numpy()
            rp = robot_asset.data.root_pos_w.detach().cpu().numpy()
            rq = robot_asset.data.root_quat_w.detach().cpu().numpy()
            traj_buffers["joint_positions"].append(jp.copy())
            traj_buffers["root_pos"].append(rp.copy())
            traj_buffers["root_quat_wxyz"].append(rq.copy())
            frames_dumped += 1

        if step % 200 == 0:
            print(f"[play_loco] step {step}/{args.steps} (dumped={frames_dumped})", flush=True)

    if traj_buffers is not None and frames_dumped > 0:
        import numpy as _np
        _np.savez(
            str(traj_path),
            joint_positions=_np.stack(traj_buffers["joint_positions"]),
            root_pos=_np.stack(traj_buffers["root_pos"]),
            root_quat_wxyz=_np.stack(traj_buffers["root_quat_wxyz"]),
            joints_list=_np.array(traj_buffers["joints_list"]),
            fps=_np.array(50.0),
        )
        print(f"[play_loco] dumped {frames_dumped} frames to {traj_path}", flush=True)

    print(f"[play_loco] playback complete after {args.steps} steps", flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
