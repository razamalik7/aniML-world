"""Is the deer in SELF-collision at baseline, and is `non_hoof_contact` measuring
ground contact at all?

Motivation: a rest-pose hull test found 13 overlapping link pairs that are NOT
parent-child, so Isaac's adjacent-link filter never covers them. Every 2-DOF joint
inserts a massless `_dof0` link, so `Pelvis -> Spine_dof0 -> Spine` makes
`Pelvis<->Spine` a GRANDPARENT pair, and the fat convex hulls of neighbouring body
segments overlap. Separately, a rollout reported 521,828 N of total contact force on
an 809 N animal, while training logged `non_hoof_contact` ~1800x smaller. One of
those is wrong and guessing is not acceptable.

This spawns the deer in a clean state, lets it settle under gravity with zero action,
and reports which bodies actually report contact -- so we can tell baseline
self-collision from a pathological collapsed pose, and check the sensor's body
ordering against physical plausibility (standing on flat ground, only hooves should
carry load).

  OMNI_KIT_ACCEPT_EULA=YES python tools/rig_pipeline/diagnose_self_collision.py
"""
import argparse, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--task", default="AniML-Deer-PolyArt-Walk-AMP-v0")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=120)
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

HOOVES = {"L_Toe0", "R_Toe0", "L_Finger0", "R_Finger0"}
NON_HOOF = ["L_Thigh", "L_Calf", "L_HorseLink", "L_Foot", "R_Thigh", "R_Calf",
            "R_HorseLink", "R_Foot", "L_UpperArm", "L_Forearm", "L_Hand",
            "R_UpperArm", "R_Forearm", "R_Hand"]

cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
for t in list(vars(cfg.terminations).keys()):
    if t != "time_out" and not t.startswith("_"):
        setattr(cfg.terminations, t, None)
cfg.episode_length_s = 10_000.0
env = gym.make(args.task, cfg=cfg, render_mode=None)
u = env.unwrapped
sensor = u.scene.sensors["contact_forces"]
robot = u.scene["robot"]
names = list(sensor.body_names)
print(f"\n[diag] sensor tracks {len(names)} bodies")
print(f"[diag] robot body order matches sensor order: "
      f"{list(robot.data.body_names) == names}")

obs, _ = env.reset()
zero = torch.zeros((args.num_envs, u.action_manager.total_action_dim), device=u.device)


def snapshot(tag):
    f = torch.linalg.norm(sensor.data.net_forces_w, dim=-1)[0]      # env 0, (bodies,)
    tot = float(f.sum())
    hoof = float(sum(f[names.index(h)] for h in HOOVES if h in names))
    nonh = float(sum(f[names.index(b)] for b in NON_HOOF if b in names))
    torso = tot - hoof - nonh
    pelvis_z = float(robot.data.root_pos_w[0, 2] - u.scene.env_origins[0, 2])
    print(f"\n--- {tag}  (pelvis z={pelvis_z:.3f}) ---")
    print(f"  TOTAL {tot:11.1f} N   hooves {hoof:9.1f}   non-hoof legs {nonh:11.1f}   "
          f"torso/other {torso:11.1f}")
    print(f"  (body weight is ~809 N; standing, hooves should carry ~all of it)")
    hot = [(names[i], float(f[i])) for i in range(len(names)) if f[i] > 1.0]
    hot.sort(key=lambda x: -x[1])
    print(f"  bodies with >1 N: {len(hot)}")
    for n, v in hot[:12]:
        kind = "HOOF" if n in HOOVES else ("leg " if n in NON_HOOF else "TORSO")
        print(f"     {kind} {n:<16}{v:12.1f} N")
    cnt1 = int(sum(1 for b in NON_HOOF if b in names and f[names.index(b)] > 1.0))
    cnt50 = int(sum(1 for b in NON_HOOF if b in names and f[names.index(b)] > 50.0))
    print(f"  non_hoof_contact would count: {cnt1} bodies at >1 N, {cnt50} at >50 N")
    return tot, hoof, nonh, torso


snapshot("t=0 (immediately after RSI spawn)")
with torch.no_grad():
    for i in range(args.steps):
        env.step(zero)
        if i in (4, 24, args.steps - 1):
            snapshot(f"t={i+1} steps (settling under gravity, zero action)")

print("\n[diag] interpretation:")
print("  * torso/other force >> 0 while standing  => genuine SELF-collision (the")
print("    grandparent pairs the adjacent-link filter misses).")
print("  * non-hoof leg force >> 0 while standing => `non_hoof_contact` is firing on")
print("    self-collision, not ground contact, and its threshold is meaningless.")
print("  * total ~= 809 N with force concentrated in hooves => rig is healthy and the")
print("    521 kN seen earlier was the collapsed off-distribution pose, not baseline.")
env.close()
app.close()
