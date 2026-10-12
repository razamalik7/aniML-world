"""Piece-2/3/4 validation gate: does the AMP walk env construct/reset/step?

Builds the real DeerPolyArtWalkAMPEnvCfg (small num_envs), resets, and steps
with zero actions. Asserts:
  * env constructs (scene/robot/managers all resolve),
  * obs groups present with expected dims: amp == 82 (38+38+3+3),
  * action dim == 34 (active joints only),
  * a few steps run finite (no NaN / crash),
  * the AMP obs joint_names amp-rsl-rl will read == REFERENCE_JOINT_ORDER.

  OMNI_KIT_ACCEPT_EULA=YES C:\\isaac-lab\\env_isaaclab\\Scripts\\python.exe -u \\
      tools/rig_pipeline/validate_env.py > env_val.log 2>&1
"""
import os, sys
from pathlib import Path
from isaaclab.app import AppLauncher

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("ANIML_DEER_USD", str(REPO / "assets/deer_polyart_urdf/deer_polyart.usd"))
os.environ["ANIML_DEER_NUM_ENVS"] = "16"

launcher = AppLauncher(headless=True)
sim_app = launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from training.polyart.amp_env_cfg import DeerPolyArtWalkAMPEnvCfg
from training.polyart.articulation import REFERENCE_JOINT_ORDER, ACTIVE_JOINTS

cfg = DeerPolyArtWalkAMPEnvCfg()
cfg.scene.num_envs = 16
env = ManagerBasedRLEnv(cfg)
print(f"ENV CONSTRUCTED  num_envs={env.num_envs}", flush=True)

obs, _ = env.reset()
pol = obs["policy"]; amp = obs["amp"]
print(f"obs policy dim = {tuple(pol.shape)}   amp dim = {tuple(amp.shape)}", flush=True)

# --- RSI spawn diagnostics (reset just ran reset_from_reference) ---
robot = env.scene["robot"]
rz = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
rlv = robot.data.root_lin_vel_w.norm(dim=-1)
rav = robot.data.root_ang_vel_w.norm(dim=-1)
bz = robot.data.body_pos_w[:, :, 2] - env.scene.env_origins[:, 2].unsqueeze(1)
feet_idx = [robot.body_names.index(b) for b in ("L_Toe0", "R_Toe0", "L_Finger0", "R_Finger0")]
min_foot = bz[:, feet_idx].min().item()
jp = robot.data.joint_pos
print(f"RSI root z: {rz.min():.3f}..{rz.max():.3f}  (expect ~0.89-0.96)", flush=True)
print(f"RSI |root_lin_vel|: {rlv.min():.2f}..{rlv.max():.2f} m/s  |root_ang_vel|: {rav.min():.2f}..{rav.max():.2f} rad/s", flush=True)
print(f"RSI min foot z: {min_foot:.3f}  joint_pos range: {jp.min():.2f}..{jp.max():.2f}", flush=True)
# second reset to confirm resampling gives a different draw (RSI pool works)
obs2, _ = env.reset()
rz2 = (robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2])
print(f"2nd RSI reset root z: {rz2.min():.3f}..{rz2.max():.3f}  finite={bool(torch.isfinite(obs2['amp']).all())}", flush=True)
print(f"AMP dim == 82 (38+38+3+3): {amp.shape[1] == 82}", flush=True)
act_dim = env.action_manager.total_action_dim
print(f"action dim = {act_dim}  (expect 34): {act_dim == 34}", flush=True)

# amp-rsl-rl reads this list to align dataset columns -> must equal REFERENCE_JOINT_ORDER
amp_names = env.cfg.observations.amp.joint_pos.params["asset_cfg"].joint_names
print(f"amp joint_names == REFERENCE_JOINT_ORDER: {list(amp_names) == list(REFERENCE_JOINT_ORDER)}", flush=True)

# step with zero actions
zero = torch.zeros((env.num_envs, act_dim), device=env.device)
finite = True
for i in range(30):
    obs, rew, term, trunc, info = env.step(zero)
    if not (torch.isfinite(obs["policy"]).all() and torch.isfinite(obs["amp"]).all() and torch.isfinite(rew).all()):
        finite = False; print(f"  NONFINITE at step {i}", flush=True); break
print(f"stepped 30x  finite={finite}  reward[mean]={rew.mean().item():.3f}  terminated={int(term.sum().item())}/{env.num_envs}", flush=True)

ok = (amp.shape[1] == 82) and (act_dim == 34) and (list(amp_names) == list(REFERENCE_JOINT_ORDER)) and finite
print("\nRESULT: " + ("PASS" if ok else "NEEDS_LOOK"), flush=True)
env.close()
sim_app.close()
