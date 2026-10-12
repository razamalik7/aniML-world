"""Do the SIM and the REFERENCE agree on what a joint angle means?

The pipeline measures joint angles in several places and they must all agree:
  * `extract_reference` -- exact twist about axis1, zero = the MODEL rest pose
  * `build_urdf`        -- rotation-vector projection, zero = animation frame 0  (this
                           one was WRONG and produced the bad joint limits)
  * the SIM             -- USD revolute-joint angles, zero = the URDF's zero pose
  * the AMP discriminator -- compares SIM joint_pos against REFERENCE joint_positions
                             column-for-column, so any offset is a permanent, invisible
                             style penalty.

Test: write a known reference frame's joint angles into the sim, read them straight
back, and compare element-wise. A per-joint constant offset means the conventions
disagree and every AMP comparison has been biased.

  OMNI_KIT_ACCEPT_EULA=YES python tools/rig_pipeline/check_angle_convention.py
"""
import argparse, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--task", default="AniML-Deer-PolyArt-Walk-AMP-v0")
parser.add_argument("--num_envs", type=int, default=8)
os.environ.setdefault("ANIML_DEER_USD", str(REPO / "assets/deer_polyart_urdf/deer_polyart.usd"))

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args, extras = parser.parse_known_args()
sys.argv = [sys.argv[0]] + extras
app = AppLauncher(headless=True).app

import gymnasium as gym
import numpy as np
import torch
import training.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
env = gym.make(args.task, cfg=cfg, render_mode=None)
u = env.unwrapped
robot = u.scene["robot"]
env.reset()

sim_names = list(robot.data.joint_names)
ref = np.load(str(REPO / "assets/motions/deer_polyart_walk.npy"), allow_pickle=True).item()
ref_names = list(ref["joints_list"])
ref_jp = np.asarray(ref["joint_positions"], float)

print(f"\n[conv] sim joints={len(sim_names)}  reference joints={len(ref_names)}")
print(f"[conv] name sets identical: {set(sim_names) == set(ref_names)}")
print(f"[conv] ORDER identical:     {sim_names == ref_names}")

col = [ref_names.index(n) for n in sim_names]        # reference col per SIM joint
dev = u.device
worst = {}
for frame in (0, 8, 16, 24, 32):
    f = min(frame, len(ref_jp) - 1)
    target = torch.tensor(ref_jp[f][col], dtype=torch.float32, device=dev).repeat(args.num_envs, 1)
    zeros = torch.zeros_like(target)
    ids = torch.arange(args.num_envs, device=dev)
    robot.write_joint_state_to_sim(target, zeros, env_ids=ids)
    u.sim.step(render=False)
    robot.update(u.sim.get_physics_dt())
    read = robot.data.joint_pos[0].detach().cpu().numpy()
    err = np.degrees(np.abs(read - ref_jp[f][col]))
    print(f"\n[conv] frame {f:>3}: max |written-readback| = {err.max():.4f} deg   "
          f"mean = {err.mean():.4f} deg")
    for i in np.argsort(err)[-3:][::-1]:
        if err[i] > 0.05:
            print(f"          {sim_names[i]:<28} wrote {np.degrees(ref_jp[f][col[i]]):8.2f}  "
                  f"read {np.degrees(read[i]):8.2f}  diff {err[i]:6.2f} deg")
        worst[sim_names[i]] = max(worst.get(sim_names[i], 0), err[i])

print("\n[conv] VERDICT:")
mx = max(worst.values()) if worst else 0.0
if mx < 0.05:
    print("  PASS - the sim reproduces reference joint angles exactly. The sim and the")
    print("  reference share one convention, so AMP is comparing like with like and the")
    print("  joint-range findings are real policy behaviour, not a measurement artefact.")
else:
    print(f"  MISMATCH - worst {mx:.2f} deg. The sim does not store the angles we write,")
    print("  so every AMP joint comparison carries a constant bias.")
env.close()
app.close()
