"""Standalone physics settle test for the Poly Art deer USD.

No training env cfg — just Isaac Lab's Articulation asset dropped on a ground
plane, held at its rest pose with DELIBERATE PD gains (not inherited magic
numbers), settled under gravity for 2 s. Reports whether it:
  * stands on its feet (pelvis Z stable near rest height),
  * stays upright (root quaternion near identity),
  * stays finite (no NaN / explosion),
  * is quiet (no self-collision jitter -> bounded joint velocity).

  OMNI_KIT_ACCEPT_EULA=YES /c/isaac-lab/env_isaaclab/Scripts/python.exe \
      tools/rig_pipeline/validate_usd.py --usd assets/deer_polyart_urdf/deer_polyart.usd
"""
import argparse
from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--usd", required=True)
ap.add_argument("--spawn_z", type=float, default=0.95)
ap.add_argument("--stiffness", type=float, default=60.0)
ap.add_argument("--damping", type=float, default=4.0)
ap.add_argument("--gravity", type=float, default=-9.81)
ap.add_argument("--self_collision", type=int, default=1)
ap.add_argument("--fix_base", type=int, default=0)
AppLauncher.add_app_launcher_args(ap)
args, _ = ap.parse_known_args()
launcher = AppLauncher(headless=True)
sim_app = launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, device="cuda:0",
                                      gravity=(0.0, 0.0, args.gravity)))
sim.set_camera_view([3.0, 3.0, 2.0], [0.0, 0.0, 0.5])

# ground + light
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
sim_utils.DomeLightCfg(intensity=2000.0).func("/World/light", sim_utils.DomeLightCfg(intensity=2000.0))

robot_cfg = ArticulationCfg(
    prim_path="/World/deer",
    spawn=sim_utils.UsdFileCfg(
        usd_path=args.usd, activate_contact_sensors=False,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=bool(args.self_collision)),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, args.spawn_z), rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={".*": 0.0}, joint_vel={".*": 0.0},
    ),
    # DELIBERATE gains: 60 N·m/rad stiffness, 4 damping — a reasonable leg PD hold,
    # NOT the inherited USD drive numbers. Chosen so I understand what's holding it.
    actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=args.stiffness, damping=args.damping)},
)
robot = Articulation(robot_cfg)

sim.reset()
# CRITICAL: write the default state to sim. Without this the joints spawn with
# uninitialized (garbage) velocity and the whole thing flails.
root_state = robot.data.default_root_state.clone()
robot.write_root_pose_to_sim(root_state[:, :7])
robot.write_root_velocity_to_sim(root_state[:, 7:])
robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
robot.reset()
print(f"BODIES:{robot.num_bodies}  JOINTS:{robot.num_joints}  gravity={args.gravity}", flush=True)
# initial pose: verify it spawns standing (feet low, head high, symmetric)
robot.update(1.0 / 120.0)
bz = robot.data.body_pos_w[0, :, 2]
names = robot.data.body_names
def show(sub):
    return {names[i]: round(bz[i].item(), 3) for i in range(len(names)) if sub in names[i]}
print("INIT feet:", show("Toe"), show("Finger"), flush=True)
print("INIT Head/Pelvis Z:", show("Head"), show("Pelvis"), flush=True)
print("INIT joint_pos |max|:", round(robot.data.joint_pos[0].abs().max().item(), 4),
      " joint_vel |max|:", round(robot.data.joint_vel[0].abs().max().item(), 4), flush=True)

# hold the rest pose
targets = torch.zeros((robot.num_instances, robot.num_joints), device=sim.device)
dt = sim.get_physics_dt()

def snap(step):
    rp = robot.data.root_pos_w[0]; rq = robot.data.root_quat_w[0]
    jv = robot.data.joint_vel[0]
    finite = bool(torch.isfinite(rp).all() and torch.isfinite(jv).all())
    # feet heights
    zs = robot.data.body_pos_w[0, :, 2]
    foot_z = [robot.data.body_names[i] + f":{zs[i].item():.3f}"
              for i in range(len(robot.data.body_names)) if ("Toe" in robot.data.body_names[i])]
    return (f"step {step:3d}  rootZ={rp[2].item():.3f}  quat=({rq[0]:.3f},{rq[1]:.3f},{rq[2]:.3f},{rq[3]:.3f})"
            f"  |jvel|max={jv.abs().max().item():.2f}  finite={finite}  toes={foot_z}")

fixed_pose = torch.tensor([[0.0, 0.0, args.spawn_z, 1.0, 0.0, 0.0, 0.0]], device=sim.device)
fixed_vel = torch.zeros((1, 6), device=sim.device)
robot.set_joint_position_target(targets)
robot.write_data_to_sim()
print("t0 " + snap(0), flush=True)
for step in range(1, 241):
    robot.set_joint_position_target(targets)
    robot.write_data_to_sim()
    sim.step()
    robot.update(dt)
    if args.fix_base:   # pin the body in the air -> isolates leg/rig health from balance
        robot.write_root_pose_to_sim(fixed_pose)
        robot.write_root_velocity_to_sim(fixed_vel)
    if step % 60 == 0:
        jpa = robot.data.joint_pos[0].abs()
        ji = jpa.argmax().item()
        # top-3 drifting joints
        top = torch.topk(jpa, 3)
        drifters = ", ".join(f"{robot.joint_names[int(i)]}={jpa[int(i)].item():.2f}" for i in top.indices)
        print(f"   {snap(step)}  drift-top3: {drifters}", flush=True)

# verdict
rp = robot.data.root_pos_w[0]; rq = robot.data.root_quat_w[0]; jv = robot.data.joint_vel[0]
upright = abs(rq[0].item()) > 0.95   # w near 1 => near-identity orientation
finite = bool(torch.isfinite(rp).all() and torch.isfinite(robot.data.joint_pos[0]).all())
quiet = jv.abs().max().item() < 2.0
stands = rp[2].item() > 0.6
print("\n=== VERDICT ===", flush=True)
print(f"final rootZ={rp[2].item():.3f} (spawn {args.spawn_z}) | upright={upright} | finite={finite} "
      f"| quiet(|jvel|<2)={quiet} (max {jv.abs().max().item():.2f}) | stands(z>0.6)={stands}", flush=True)
print("RESULT: " + ("PASS" if (finite and upright and stands and quiet) else "NEEDS_LOOK"), flush=True)
sim_app.close()
