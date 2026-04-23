"""Spawn 1 deer env using the actual DeerUsdFlatEnvCfg, settle, print body positions.

Usage:
  _isaac_sim\\python.bat tools/diagnose_spawn.py

Reports pelvis and foot Z heights so you can set ANIML_DEER_SPAWN_Z correctly.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("ANIML_DEER_ACTIVATE_CONTACT_SENSORS", "0")
os.environ.setdefault("ANIML_DEER_BASE_CONTACT_THRESHOLD", "0")

from isaaclab.app import AppLauncher  # noqa: E402

launcher = AppLauncher(headless=True, num_envs=1)
sim = launcher.app

import torch  # noqa: E402

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from training.envs.deer_usd_rough_env_cfg import DeerUsdFlatEnvCfg  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

cfg = DeerUsdFlatEnvCfg()
cfg.scene.num_envs = 1

env = ManagerBasedRLEnv(cfg)
robot = env.scene["robot"]

body_names = robot.body_names
joint_names = robot.joint_names
print(f"\nBody names ({len(body_names)}): {body_names}", flush=True)
print(f"Joint names ({len(joint_names)}): {joint_names}", flush=True)

obs, _ = env.reset()

print("\n--- After env.reset() (timestep 0) ---", flush=True)
body_pos = robot.data.body_pos_w[0]
root_quat = robot.data.root_quat_w[0]
for i, name in enumerate(body_names):
    p = body_pos[i]
    marker = ""
    if "foot" in name.lower():
        marker = " <-- FOOT"
    elif "pelvis" in name.lower():
        marker = " <-- PELVIS"
    print(f"  {name:20s}  X={p[0].item():.4f}  Y={p[1].item():.4f}  Z={p[2].item():.4f}{marker}", flush=True)

root_p = robot.data.root_pos_w[0]
print(f"\n  Root pos:  {root_p[0].item():.4f}, {root_p[1].item():.4f}, {root_p[2].item():.4f}", flush=True)
print(f"  Root quat: {root_quat[0].item():.4f}, {root_quat[1].item():.4f}, {root_quat[2].item():.4f}, {root_quat[3].item():.4f}", flush=True)

print("\n--- Settling for 240 steps (2 seconds at dt=1/120) with zero actions ---", flush=True)
zero_act = torch.zeros(1, env.action_space.shape[-1], device=env.device)
for step in range(240):
    obs, rew, terminated, truncated, info = env.step(zero_act)
    if step % 60 == 59:
        bp = robot.data.body_pos_w[0]
        rp = robot.data.root_pos_w[0]
        rq = robot.data.root_quat_w[0]
        print(f"\n  Step {step+1}:", flush=True)
        for i, name in enumerate(body_names):
            p = bp[i]
            marker = ""
            if "foot" in name.lower():
                marker = " <-- FOOT"
            elif "pelvis" in name.lower():
                marker = " <-- PELVIS"
            print(f"    {name:20s}  Z={p[2].item():.4f}{marker}", flush=True)
        print(f"    Root pos Z={rp[2].item():.4f}  quat=({rq[0].item():.4f},{rq[1].item():.4f},{rq[2].item():.4f},{rq[3].item():.4f})", flush=True)

body_pos = robot.data.body_pos_w[0]
print("\n\n=== FINAL STATE (after 2s settle) ===", flush=True)
pelvis_z = None
foot_data = []
for i, name in enumerate(body_names):
    p = body_pos[i]
    z = p[2].item()
    marker = ""
    if "foot" in name.lower():
        marker = " <-- FOOT"
        foot_data.append((name, z))
    elif "pelvis" in name.lower():
        marker = " <-- PELVIS"
        pelvis_z = z
    print(f"  {name:20s}  X={p[0].item():.4f}  Y={p[1].item():.4f}  Z={p[2].item():.4f}{marker}", flush=True)

print(f"\n  Pelvis Z = {pelvis_z:.4f}" if pelvis_z is not None else "  No pelvis found", flush=True)
for fn, fz in foot_data:
    print(f"  {fn} Z = {fz:.4f}", flush=True)

if pelvis_z is not None and foot_data:
    min_foot = min(z for _, z in foot_data)
    max_foot = max(z for _, z in foot_data)
    print(f"\n  Min foot Z: {min_foot:.4f}", flush=True)
    print(f"  Max foot Z: {max_foot:.4f}", flush=True)
    if pelvis_z < 0.05:
        print("  >> COLLAPSED: Pelvis on ground", flush=True)
    elif all(z > pelvis_z for _, z in foot_data):
        print("  >> UPSIDE DOWN: Feet above pelvis", flush=True)
    elif min_foot < -0.05:
        print(f"  >> SINKING: Feet below ground by {abs(min_foot):.3f}m", flush=True)
    elif min_foot > 0.1:
        spawn_z = float(os.environ.get("ANIML_DEER_SPAWN_Z", "0.55"))
        new_z = spawn_z - min_foot + 0.01
        print(f"  >> FLOATING: Lower spawn Z. Try ANIML_DEER_SPAWN_Z={new_z:.3f}", flush=True)
    else:
        print(f"  >> LOOKS GOOD: Feet near ground, pelvis elevated at {pelvis_z:.3f}", flush=True)
        print(f"  >> Recommended ANIML_DEER_SPAWN_Z={pelvis_z:.3f}", flush=True)

env.close()
