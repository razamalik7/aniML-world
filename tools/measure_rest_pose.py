"""
Spawn the deer, freeze physics for 1 step, print all joint positions and the
root height. This gives us the ground-truth rest angles in Isaac's frame.
"""
import os, math
os.environ["ANIML_DEER_USD"]   = r"C:\Users\nars7\Downloads\aniML_world\deer_world.usd"
os.environ["ANIML_DEER_SPAWN_Z"] = "0.64"
os.environ["ANIML_DEER_SPAWN_ROT"] = "1,0,0,0"
os.environ["ANIML_DEER_STIFFNESS"] = "0"   # no actuator force — pure physics
os.environ["ANIML_DEER_DAMPING"]   = "0"
os.environ["ANIML_DEER_BASE_CONTACT_THRESHOLD"] = "0"
os.environ["PYTHONPATH"] = r"C:\Users\nars7\Downloads\aniML_world"

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import gymnasium as gym
import torch
import sys
sys.path.insert(0, r"C:\Users\nars7\Downloads\aniML_world")

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnv
import training.envs  # noqa: F401

env_cfg_module = __import__(
    "training.envs.deer_usd_rough_env_cfg", fromlist=["DeerUsdFlatEnvCfg"]
)
EnvCfg = env_cfg_module.DeerUsdFlatEnvCfg
cfg = EnvCfg()
cfg.scene.num_envs = 1
cfg.sim.device = "cuda:0"

env = ManagerBasedRLEnv(cfg=cfg)

# Step once with zero action to let physics settle for 1 frame
obs, _ = env.reset()
zero_act = torch.zeros(env.action_space.shape, device="cuda:0")
obs, rew, done, trunc, info = env.step(zero_act)

robot = env.scene["robot"]

LOG = open(r"C:\Users\nars7\Downloads\aniML_world\tools\rest_pose_log.txt", "w")

# Root height
root_pos = robot.data.root_pos_w[0]
LOG.write(f"Root (pelvis) world Z at frame 1: {root_pos[2].item():.4f}\n\n")

# Joint positions
joint_names = robot.data.joint_names
joint_pos   = robot.data.joint_pos[0]
LOG.write("Joint positions at frame 1 (these are the natural rest angles):\n")
for name, pos in zip(joint_names, joint_pos):
    LOG.write(f"  {name}: {pos.item():.4f} rad ({math.degrees(pos.item()):.1f} deg)\n")

# Root orientation (quaternion w,x,y,z)
root_quat = robot.data.root_quat_w[0]
LOG.write(f"\nRoot quaternion (w,x,y,z): {root_quat.tolist()}\n")

LOG.close()
env.close()
app.close()
print("Done. See tools/rest_pose_log.txt")
