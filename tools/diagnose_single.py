"""Single-test diagnostic: spawn deer, step physics, print link Z values.

Usage:
  isaaclab.bat -p tools/diagnose_single.py --rot_deg 0
  isaaclab.bat -p tools/diagnose_single.py --rot_deg -91
  isaaclab.bat -p tools/diagnose_single.py --rot_deg 90
  isaaclab.bat -p tools/diagnose_single.py --rot_deg 160
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import DCMotorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

USD_PATH = str(ROOT / "deer_standing_robot.usd")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rot_deg", type=float, default=0.0,
                        help="X-axis rotation in degrees")
    parser.add_argument("--spawn_z", type=float, default=2.0,
                        help="Spawn height")
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    r = math.radians(args.rot_deg)
    qw, qx = math.cos(r / 2), math.sin(r / 2)

    cfg = ArticulationCfg(
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
            pos=(0.0, 0.0, args.spawn_z),
            rot=(qw, qx, 0.0, 0.0),
            joint_pos={},
        ),
        actuators={
            "legs": DCMotorCfg(
                joint_names_expr=[".*"],
                saturation_effort=350, effort_limit=300,
                velocity_limit=100,
                stiffness={".*": 200}, damping={".*": 6},
            )
        },
    )

    sim = SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device="cuda:0"))
    robot = Articulation(cfg)
    sim.reset()
    robot.reset()

    def dump(tag):
        robot.update(dt=sim.get_physics_dt())
        pos = robot.data.body_pos_w[0].cpu().numpy()
        names = robot.body_names
        print(f"\n--- {tag} (rot={args.rot_deg}°, z={args.spawn_z}) ---")
        for i, n in enumerate(names):
            print(f"  {n:15s} x={pos[i,0]:+8.4f} y={pos[i,1]:+8.4f} z={pos[i,2]:+8.4f}")

        feet = ["HL_foot", "HR_foot", "FL_foot", "FR_foot"]
        foot_zs = [pos[names.index(f), 2] for f in feet if f in names]
        pelvis_z = pos[names.index("pelvis"), 2] if "pelvis" in names else 0
        skull_z = pos[names.index("skull"), 2] if "skull" in names else 0
        print(f"\n  Pelvis Z:       {pelvis_z:.4f}")
        print(f"  Skull Z:        {skull_z:.4f}")
        print(f"  Mean foot Z:    {sum(foot_zs)/len(foot_zs):.4f}")
        print(f"  Pelvis-feet:    {pelvis_z - sum(foot_zs)/len(foot_zs):.4f}")
        print(f"  Min foot Z:     {min(foot_zs):.4f}")
        print(f"  Max foot Z:     {max(foot_zs):.4f}")

    dump("After reset (step 0)")

    for i in range(args.steps):
        robot.set_joint_position_target(robot.data.default_joint_pos)
        robot.write_data_to_sim()
        sim.step()

    dump(f"After {args.steps} steps")

    app.close()


if __name__ == "__main__":
    main()
