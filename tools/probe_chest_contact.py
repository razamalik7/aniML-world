"""Spawn the deer in default pose for 100 sim steps, print the contact force
on each torso body each step. Diagnoses why base_contact termination fires
on chest at threshold=200N: is it real ground contact, or self-collision noise?
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=120)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args)
sim_app = app.app

import gymnasium as gym
import torch

import training.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


task = "AniML-Deer-USD-Loco-Stage1-v0"
env_cfg = parse_env_cfg(task, device=args.device, num_envs=4)
env = gym.make(task, cfg=env_cfg)
isaac_env = env.unwrapped

# Find the contact sensor + body indices for chest/neck/skull
sensor = isaac_env.scene.sensors["contact_forces"]
all_body_names = sensor.body_names
print(f"contact_forces body_names ({len(all_body_names)}):")
for i, n in enumerate(all_body_names):
    print(f"  [{i:2d}] {n}")

torso_names = ["chest", "neck", "skull"]
torso_idx = [i for i, n in enumerate(all_body_names) if n in torso_names]
print(f"torso indices: {torso_idx}")

# Step a few times and dump forces.
obs, _ = env.reset()
zero_action = torch.zeros(env.action_space.shape, device=args.device)
print(f"\nstep, env_id, body, |force_world| (N)")
for step in range(args.steps):
    obs, _, _, _, _ = env.step(zero_action)
    if step % 10 == 0:
        forces = sensor.data.net_forces_w  # (num_envs, num_bodies, 3)
        for env_id in range(min(2, forces.shape[0])):
            for bi in torso_idx:
                f = forces[env_id, bi]
                mag = float(torch.norm(f))
                if mag > 0.5:
                    print(f"  step={step:3d} env={env_id} body={all_body_names[bi]:8s} |F|={mag:8.2f}")

env.close()
sim_app.close()
