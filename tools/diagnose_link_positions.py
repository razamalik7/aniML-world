"""Diagnose deer link world positions at various root rotations.

Spawns the deer articulation, steps physics a few times, and prints
the world (x, y, z) of every link.  No RL, no rewards — just physics.

Run:
    python tools/diagnose_link_positions.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omni.isaac.core.world import World  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import DCMotorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402


def quat_from_x_deg(deg: float) -> tuple[float, float, float, float]:
    """Return (w, x, y, z) quaternion for rotation of `deg` degrees about +X."""
    r = math.radians(deg)
    return (math.cos(r / 2), math.sin(r / 2), 0.0, 0.0)


USD_PATH = str(ROOT / "deer_standing_robot.usd")

BODY_NAMES = [
    "pelvis", "chest", "neck", "skull",
    "HL_thigh", "HL_shin", "HL_foot",
    "HR_thigh", "HR_shin", "HR_foot",
    "FL_thigh", "FL_shin", "FL_foot",
    "FR_thigh", "FR_shin", "FR_foot",
]

IK_JOINT_POS = {
    "pelvis_to_chest": 0.49,
    "pelvis_to_HR_thigh": -0.4591,
    "HR_thigh_to_HR_shin": -0.4644,
    "pelvis_to_HL_thigh": -0.4591,
    "HL_thigh_to_HL_shin": -0.4644,
    "chest_to_FR_thigh": 0.3229,
    "FR_thigh_to_FR_shin": 0.1745,
    "chest_to_FL_thigh": 0.3356,
    "FL_thigh_to_FL_shin": 0.1745,
}


def make_cfg(spawn_z: float, rot: tuple, joint_pos: dict) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, spawn_z),
            rot=rot,
            joint_pos=joint_pos,
        ),
        actuators={
            "legs": DCMotorCfg(
                joint_names_expr=[".*"],
                saturation_effort=350,
                effort_limit=300,
                velocity_limit=100,
                stiffness={".*": 200},
                damping={".*": 6},
            )
        },
    )


def print_link_positions(robot: Articulation, label: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    all_names = robot.body_names
    pos_w = robot.data.body_pos_w[0]  # shape (num_bodies, 3)
    for name in BODY_NAMES:
        if name in all_names:
            idx = all_names.index(name)
            p = pos_w[idx].cpu().numpy()
            print(f"  {name:15s}  x={p[0]:+8.4f}  y={p[1]:+8.4f}  z={p[2]:+8.4f}")
        else:
            print(f"  {name:15s}  NOT FOUND")

    foot_names = ["HL_foot", "HR_foot", "FL_foot", "FR_foot"]
    foot_zs = []
    for fn in foot_names:
        if fn in all_names:
            foot_zs.append(pos_w[all_names.index(fn), 2].item())
    if foot_zs:
        mean_z = np.mean(foot_zs)
        pelvis_z = pos_w[all_names.index("pelvis"), 2].item() if "pelvis" in all_names else 0
        print(f"\n  Foot Zs: {[f'{z:.4f}' for z in foot_zs]}")
        print(f"  Mean foot Z:     {mean_z:.4f}")
        print(f"  Pelvis Z:        {pelvis_z:.4f}")
        print(f"  Pelvis - feet:   {pelvis_z - mean_z:.4f}  (should be >0.2 for standing)")
        skull_z = pos_w[all_names.index("skull"), 2].item() if "skull" in all_names else 0
        print(f"  Skull Z:         {skull_z:.4f}")
        print(f"  Skull - pelvis:  {skull_z - pelvis_z:.4f}")


def run_test(label: str, spawn_z: float, rot: tuple, joint_pos: dict, steps: int = 50):
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device="cuda:0"))
    sim.set_camera_view(eye=(3.0, 3.0, 3.0), target=(0.0, 0.0, 0.5))
    cfg = make_cfg(spawn_z, rot, joint_pos)
    robot = Articulation(cfg)
    sim.reset()
    robot.reset()
    print(f"\n--- After reset (0 steps) ---")
    robot.update(dt=sim.get_physics_dt())
    print_link_positions(robot, f"{label} — step 0")

    for i in range(steps):
        zero_actions = torch.zeros(1, robot.num_joints, device=robot.device)
        robot.set_joint_position_target(
            robot.data.default_joint_pos + zero_actions
        )
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt=sim.get_physics_dt())

    print_link_positions(robot, f"{label} — step {steps}")
    sim.clear_all_callbacks()
    sim._app_control_manager = None


def main():
    lines = []

    # Test 1: Native USD orient (-91° X) with zero joint angles, spawn at z=1.0
    native_quat = (0.7018, -0.7124, 0.0, 0.0)
    print("\n" + "#" * 70)
    print("TEST 1: Native -91° rotation, zero joints, z=1.0")
    print("#" * 70)
    run_test("Native -91°, zero joints, z=1.0", spawn_z=1.0, rot=native_quat, joint_pos={})

    # Test 2: 160° rotation with IK joints, spawn at z=0.34
    rot_160 = quat_from_x_deg(160.0)
    print("\n" + "#" * 70)
    print("TEST 2: 160° rotation, IK joints, z=0.34")
    print("#" * 70)
    run_test("160° + IK joints, z=0.34", spawn_z=0.34, rot=rot_160, joint_pos=IK_JOINT_POS)

    # Test 3: Identity rotation, zero joints, z=1.0
    print("\n" + "#" * 70)
    print("TEST 3: Identity rotation, zero joints, z=1.0")
    print("#" * 70)
    run_test("Identity, zero joints, z=1.0", spawn_z=1.0, rot=(1.0, 0.0, 0.0, 0.0), joint_pos={})

    # Test 4: 160° rotation, IK joints, z=1.0 (higher spawn)
    print("\n" + "#" * 70)
    print("TEST 4: 160° rotation, IK joints, z=1.0 (high spawn)")
    print("#" * 70)
    run_test("160° + IK joints, z=1.0", spawn_z=1.0, rot=rot_160, joint_pos=IK_JOINT_POS)

    app.close()


if __name__ == "__main__":
    main()
