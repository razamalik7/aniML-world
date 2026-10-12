"""Zero-dependency playback for the r38 AMP checkpoint.

The AMP runner (amp_rsl_rl) needs rsl_rl 3.0's legacy ActorCritic, which
conflicts with the rsl_rl 5.x the loco path needs. Rather than downgrade the
env or install onnxruntime, we reconstruct the actor MLP directly from the
checkpoint's model_state_dict (a plain 57->512->256->128->15 ELU MLP, no obs
normalizer) and run deterministic (mean-action) inference. Dumps the same NPZ
format as play_loco.py for tools/render_playback_meshed.py.

Usage:
    python tools/play_r38_reconstruct.py \
        --task AniML-Deer-USD-Walk-AMP-v29 \
        --checkpoint <abs>/model_8996.pt
"""
from __future__ import annotations

# h5py stub (Isaac Sim hdf5.dll conflict) — must precede any isaaclab import.
import sys, types as _types, importlib.machinery as _machinery
_stub = _types.ModuleType("h5py")
_stub.__spec__ = _machinery.ModuleSpec("h5py", loader=None)
_stub.__version__ = "3.16.0-stubbed"
class _E:
    def __getattr__(self, n): raise AttributeError(f"h5py.{n} stubbed")
_stub.__getattr__ = _E().__getattr__
_stub.File = _E().__getattr__
_stub.Group = type("Group", (), {})
sys.modules["h5py"] = _stub
_es = _types.ModuleType("h5py._errors"); _es.__spec__ = _machinery.ModuleSpec("h5py._errors", loader=None)
sys.modules["h5py._errors"] = _es

import argparse
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task", default="AniML-Deer-USD-Walk-AMP-v29")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--video_length", type=int, default=1500)
    parser.add_argument("--steps", type=int, default=1600)

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args, extras = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + extras
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import numpy as np
    import torch
    import torch.nn as nn

    import training.tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg

    # --- reconstruct actor from checkpoint (plain ELU MLP, no obs norm) ---
    ck = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    sd = ck["model_state_dict"]
    dims = [sd["actor.0.weight"].shape[1], 512, 256, 128, sd["actor.6.weight"].shape[0]]
    print(f"[r38] actor dims: {dims} (obs={dims[0]}, act={dims[-1]})", flush=True)
    actor = nn.Sequential(
        nn.Linear(dims[0], 512), nn.ELU(),
        nn.Linear(512, 256), nn.ELU(),
        nn.Linear(256, 128), nn.ELU(),
        nn.Linear(128, dims[-1]),
    ).to(args.device)
    actor.load_state_dict({k[len("actor."):]: v for k, v in sd.items() if k.startswith("actor.")})
    actor.eval()

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)
    vec_env = RslRlVecEnvWrapper(env.unwrapped)  # type: ignore[arg-type]

    isaac_env = env.unwrapped
    robot = isaac_env.scene["robot"]
    joints = list(robot.data.joint_names)

    def policy_obs():
        # Pull the policy obs group straight from the observation manager so we
        # don't depend on the wrapper's get_observations()/step() return format.
        d = isaac_env.observation_manager.compute()
        o = d["policy"] if isinstance(d, dict) else d
        return o.to(args.device)

    buffers = {"joint_positions": [], "root_pos": [], "root_quat_wxyz": []}
    obs = policy_obs()
    print(f"[r38] policy obs shape: {tuple(obs.shape)} (expect (N,{dims[0]}))", flush=True)
    assert obs.shape[-1] == dims[0], f"obs dim {obs.shape[-1]} != actor obs {dims[0]}"
    dumped = 0
    for step in range(args.steps):
        with torch.inference_mode():
            act = actor(obs)                     # deterministic mean action
        vec_env.step(act.to(vec_env.device))
        obs = policy_obs()
        if dumped < args.video_length:
            buffers["joint_positions"].append(robot.data.joint_pos.detach().cpu().numpy().copy())
            buffers["root_pos"].append(robot.data.root_pos_w.detach().cpu().numpy().copy())
            buffers["root_quat_wxyz"].append(robot.data.root_quat_w.detach().cpu().numpy().copy())
            dumped += 1
        if step % 200 == 0:
            print(f"[r38] step {step}/{args.steps} (dumped={dumped})", flush=True)

    from datetime import datetime as _dt
    out = _repo_root() / "logs" / "videos" / f"play_r38_{_dt.now().strftime('%Y-%m-%d_%H-%M-%S')}.npz"
    np.savez(str(out),
             joint_positions=np.stack(buffers["joint_positions"]),
             root_pos=np.stack(buffers["root_pos"]),
             root_quat_wxyz=np.stack(buffers["root_quat_wxyz"]),
             joints_list=np.array(joints), fps=np.array(50.0))
    print(f"[r38] dumped {dumped} frames to {out}", flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
