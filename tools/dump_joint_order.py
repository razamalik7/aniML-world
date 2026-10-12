"""Dump the runtime joint order from the v4 USD as Isaac Lab sees it."""
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "experience": r"C:\isaac-lab\IsaacLab\apps\isaaclab.python.headless.kit"})

from pathlib import Path
import os, sys
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.environ["ANIML_DEER_USD"] = str(REPO / "deer_world.usd")
os.environ["ANIML_DEER_ACTIVATE_CONTACT_SENSORS"] = "1"

import gymnasium as gym
import training.tasks  # registration
from isaaclab_tasks.utils import parse_env_cfg

env_cfg = parse_env_cfg("AniML-Deer-USD-Loco-Stage1-v0", device="cuda:0", num_envs=1)
env = gym.make("AniML-Deer-USD-Loco-Stage1-v0", cfg=env_cfg)
robot = env.unwrapped.scene["robot"]
joints = list(robot.data.joint_names)

with open(r"C:\Users\nars7\AppData\Local\Temp\joint_order.txt", "w") as f:
    f.write(f"num joints: {len(joints)}\n")
    for i, n in enumerate(joints):
        f.write(f"  {i:3d}: {n}\n")

env.close()
print("[dump] done", flush=True)
app.close()
