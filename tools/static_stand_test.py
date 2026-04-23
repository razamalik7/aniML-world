"""Static standing test: can the deer physically stand with PD holding default angles?

Spawns 1 deer at the exact default pose (no joint noise), sends zero actions
(PD tracks default joint positions), and monitors for 1000 steps (~8 seconds).

Logs: pelvis Z, foot Z, joint torques vs effort limits, saturation.

Usage (from IsaacLab root):
  isaaclab.bat -p <repo>/tools/static_stand_test.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("ANIML_DEER_ACTIVATE_CONTACT_SENSORS", "1")
os.environ.setdefault("ANIML_DEER_BASE_CONTACT_THRESHOLD", "0")

from isaaclab.app import AppLauncher

launcher = AppLauncher(headless=True, num_envs=1)
sim = launcher.app

import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from training.envs.deer_usd_rough_env_cfg import DeerUsdFlatEnvCfg
from isaaclab.envs import ManagerBasedRLEnv

cfg = DeerUsdFlatEnvCfg()
cfg.scene.num_envs = 1

env = ManagerBasedRLEnv(cfg)
robot = env.scene["robot"]

body_names = robot.body_names
joint_names = robot.joint_names
print(f"\nBodies ({len(body_names)}): {body_names}", flush=True)
print(f"Joints ({len(joint_names)}): {joint_names}", flush=True)

obs, _ = env.reset()

# Force joints to exact default positions (undo the 0.95-1.05 reset noise)
robot.write_joint_state_to_sim(
    robot.data.default_joint_pos.clone(),
    torch.zeros_like(robot.data.default_joint_vel),
)

effort_limit = float(os.environ.get("ANIML_DEER_EFFORT_LIMIT", "300"))
stiffness = float(os.environ.get("ANIML_DEER_STIFFNESS", "200"))
damping = float(os.environ.get("ANIML_DEER_DAMPING", "6"))
print(f"\nActuator config: stiffness={stiffness}, damping={damping}, effort_limit={effort_limit}")
print(f"Total mass: {robot.data.default_mass.sum().item():.2f} kg")
print(f"Weight force: {robot.data.default_mass.sum().item() * 9.81:.1f} N")

# Per-link masses
print("\nPer-link masses:")
for i, name in enumerate(body_names):
    m = robot.data.default_mass[0, i].item()
    print(f"  {name:20s}  {m:.3f} kg")

print(f"\nDefault joint positions:")
for i, name in enumerate(joint_names):
    pos = robot.data.default_joint_pos[0, i].item()
    print(f"  {name:30s}  {pos:.4f} rad  ({pos * 57.2958:.1f} deg)")

# Initial body positions
print("\n--- Initial state (t=0) ---")
bp = robot.data.body_pos_w[0]
for i, name in enumerate(body_names):
    p = bp[i]
    tag = ""
    if "foot" in name.lower():
        tag = " <-- FOOT"
    elif "pelvis" in name.lower():
        tag = " <-- PELVIS"
    print(f"  {name:20s}  Z={p[2].item():.4f}{tag}")

rq = robot.data.root_quat_w[0]
print(f"  Root quat: w={rq[0]:.4f} x={rq[1]:.4f} y={rq[2]:.4f} z={rq[3]:.4f}")

NUM_STEPS = 1000
REPORT_EVERY = 100
zero_act = torch.zeros(1, env.action_space.shape[-1], device=env.device)

pelvis_z_history = []
foot_z_history = []
max_torque_ratio_history = []

print(f"\n{'='*80}")
print(f"Running {NUM_STEPS} steps with zero actions (PD holding default pose)...")
print(f"{'='*80}\n")

for step in range(NUM_STEPS):
    obs, rew, terminated, truncated, info = env.step(zero_act)

    bp = robot.data.body_pos_w[0]
    pelvis_idx = body_names.index("pelvis") if "pelvis" in body_names else 0
    pelvis_z = bp[pelvis_idx, 2].item()
    pelvis_z_history.append(pelvis_z)

    foot_zs = []
    for i, name in enumerate(body_names):
        if "foot" in name.lower():
            foot_zs.append(bp[i, 2].item())
    foot_z_history.append(foot_zs)

    # Joint torques: applied_torque is what the actuator actually sends
    torques = robot.data.applied_torque[0]
    torque_ratio = (torques.abs() / effort_limit).max().item()
    max_torque_ratio_history.append(torque_ratio)

    if (step + 1) % REPORT_EVERY == 0 or step == 0:
        print(f"\n--- Step {step + 1}/{NUM_STEPS} ---")
        print(f"  Pelvis Z:  {pelvis_z:.4f} m")

        for i, name in enumerate(body_names):
            if "foot" in name.lower():
                print(f"  {name:15s} Z: {bp[i, 2].item():.4f} m")

        print(f"\n  Joint torques (applied / limit = {effort_limit}):")
        for i, name in enumerate(joint_names):
            t = torques[i].item()
            ratio = abs(t) / effort_limit
            saturated = " ** SATURATED **" if ratio > 0.95 else ""
            print(f"    {name:30s}  torque={t:8.2f} N·m  ({ratio*100:5.1f}%){saturated}")

        # Joint position drift from default
        cur_pos = robot.data.joint_pos[0]
        def_pos = robot.data.default_joint_pos[0]
        drift = (cur_pos - def_pos).abs()
        print(f"\n  Joint position drift from default:")
        for i, name in enumerate(joint_names):
            d = drift[i].item()
            c = cur_pos[i].item()
            print(f"    {name:30s}  current={c:.4f}  drift={d:.4f} rad ({d*57.2958:.1f} deg)")

        rq = robot.data.root_quat_w[0]
        rp = robot.data.root_pos_w[0]
        print(f"\n  Root pos: ({rp[0]:.4f}, {rp[1]:.4f}, {rp[2]:.4f})")
        print(f"  Root quat: w={rq[0]:.4f} x={rq[1]:.4f} y={rq[2]:.4f} z={rq[3]:.4f}")

# Final verdict
print(f"\n{'='*80}")
print("FINAL VERDICT")
print(f"{'='*80}")

initial_pelvis_z = pelvis_z_history[0]
final_pelvis_z = pelvis_z_history[-1]
min_pelvis_z = min(pelvis_z_history)
max_pelvis_z = max(pelvis_z_history)
pelvis_drop = initial_pelvis_z - final_pelvis_z

print(f"\n  Pelvis Z: initial={initial_pelvis_z:.4f}, final={final_pelvis_z:.4f}, "
      f"min={min_pelvis_z:.4f}, max={max_pelvis_z:.4f}")
print(f"  Pelvis drop: {pelvis_drop:.4f} m ({pelvis_drop*100:.1f} cm)")

final_feet = foot_z_history[-1]
print(f"  Final foot Z values: {[f'{z:.4f}' for z in final_feet]}")
print(f"  Max torque ratio seen: {max(max_torque_ratio_history)*100:.1f}%")

any_saturated = any(r > 0.95 for r in max_torque_ratio_history)
sat_str = "YES" if any_saturated else "NO"
print(f"  Any joint ever saturated (>95% effort): {sat_str}")

if final_pelvis_z < 0.10:
    print("\n  >> COLLAPSED: Deer cannot physically stand. The PD controller")
    print("     cannot hold the default pose against gravity.")
    print("     ROOT CAUSE: Physical model problem (actuators, mass, collision geometry).")
elif pelvis_drop > 0.10:
    print(f"\n  >> UNSTABLE: Pelvis dropped {pelvis_drop:.3f}m. Deer is slowly collapsing.")
    print("     The PD controller is too weak to maintain the pose long-term.")
elif final_pelvis_z > 0.20 and pelvis_drop < 0.05:
    print(f"\n  >> STANDING: Deer holds pose! Pelvis stable at {final_pelvis_z:.3f}m.")
    print("     The physical model is fine — problem is in RL reward/exploration.")
else:
    print(f"\n  >> MARGINAL: Pelvis at {final_pelvis_z:.3f}m, dropped {pelvis_drop:.3f}m.")
    print("     Borderline — may need stronger actuators.")

env.close()
