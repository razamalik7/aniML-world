"""Piece-1 validation gate: does Isaac Lab accept deer_polyart_articulation_cfg?

Spawns the real ArticulationCfg from training/envs/deer_polyart_articulation.py
(NOT a hand-written one) on a ground plane and asserts:
  * 38 joints, names exactly the expected partition,
  * each of the 5 actuator groups wrote its deliberate stiffness/damping to sim,
  * spawns finite & standing (a few settle steps).

  OMNI_KIT_ACCEPT_EULA=YES C:\\isaac-lab\\env_isaaclab\\Scripts\\python.exe \\
      tools/rig_pipeline/validate_articulation.py
"""
import os, sys
from pathlib import Path
from isaaclab.app import AppLauncher

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("ANIML_DEER_USD", str(REPO / "assets/deer_polyart_urdf/deer_polyart.usd"))

launcher = AppLauncher(headless=True)
sim_app = launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.assets import Articulation
from training.polyart.articulation import (
    deer_polyart_articulation_cfg, deer_polyart_actuators,
    HIND_LEG_JOINTS, FRONT_LEG_JOINTS, SPINE_JOINTS, NECK_HEAD_JOINTS, COSMETIC_JOINTS, ALL_JOINTS,
)

sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, device="cuda:0"))
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
sim_utils.DomeLightCfg(intensity=2000.0).func("/World/light", sim_utils.DomeLightCfg(intensity=2000.0))

cfg = deer_polyart_articulation_cfg()
cfg.prim_path = "/World/deer"
robot = Articulation(cfg)
sim.reset()
robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])
robot.write_root_velocity_to_sim(robot.data.default_root_state[:, 7:])
robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
robot.reset()

names = list(robot.joint_names)
print(f"NUM_JOINTS: {robot.num_joints}  NUM_BODIES: {robot.num_bodies}", flush=True)
ok_names = set(names) == set(ALL_JOINTS)
print(f"JOINT_NAMES match expected 38: {ok_names}", flush=True)
if not ok_names:
    print("  extra:", sorted(set(names) - set(ALL_JOINTS)))
    print("  missing:", sorted(set(ALL_JOINTS) - set(names)))

# per-group gain verification: read what actually landed in sim
stiff = robot.data.joint_stiffness[0]
damp = robot.data.joint_damping[0]
idx = {n: i for i, n in enumerate(names)}
acts = deer_polyart_actuators()
groups = {
    "hind_legs": HIND_LEG_JOINTS, "front_legs": FRONT_LEG_JOINTS,
    "spine": SPINE_JOINTS, "neck_head": NECK_HEAD_JOINTS, "cosmetic": COSMETIC_JOINTS,
}
all_gains_ok = True
for g, jlist in groups.items():
    want_k = acts[g].stiffness
    want_d = acts[g].damping
    got_k = [round(stiff[idx[j]].item(), 2) for j in jlist]
    got_d = [round(damp[idx[j]].item(), 2) for j in jlist]
    k_ok = all(abs(v - want_k) < 1e-3 for v in got_k)
    d_ok = all(abs(v - want_d) < 1e-3 for v in got_d)
    all_gains_ok &= (k_ok and d_ok)
    print(f"  [{g:10s}] n={len(jlist):2d}  stiffness->{want_k} ok={k_ok}  damping->{want_d} ok={d_ok}", flush=True)

# spawn sanity: a few settle steps holding the rest pose
targets = torch.zeros((robot.num_instances, robot.num_joints), device=sim.device)
for _ in range(120):
    robot.set_joint_position_target(targets)
    robot.write_data_to_sim()
    sim.step()
    robot.update(1.0 / 120.0)
rp = robot.data.root_pos_w[0]
jv = robot.data.joint_vel[0]
finite = bool(torch.isfinite(rp).all() and torch.isfinite(robot.data.joint_pos[0]).all())
bz = robot.data.body_pos_w[0, :, 2]
feet = {names_b: round(bz[i].item(), 3) for i, names_b in enumerate(robot.body_names) if names_b in ("L_Toe0", "R_Toe0", "L_Finger0", "R_Finger0")}
print(f"\nafter 1s settle: rootZ={rp[2].item():.3f}  finite={finite}  |jvel|max={jv.abs().max().item():.2f}", flush=True)
print(f"feet Z: {feet}", flush=True)
print("\nRESULT: " + ("PASS" if (ok_names and all_gains_ok and finite) else "NEEDS_LOOK"), flush=True)
sim_app.close()
