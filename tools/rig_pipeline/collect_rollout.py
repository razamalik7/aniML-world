"""Roll out a trained polyart AMP policy and dump a trajectory .npz for
tools/render_playback_meshed.py (which FK-renders the real mesh, bypassing
Isaac's renderer).

Terminations are disabled and the episode made effectively infinite so the clip
is continuous (you see the raw sustained behavior — stand, shuffle, walk, or
fall — without reset jumps). Env 0 gets an RSI start; to also show a cold stand
start, pass --stand_start to reset joints to zero after the RSI reset.

  OMNI_KIT_ACCEPT_EULA=YES C:\\isaac-lab\\env_isaaclab\\Scripts\\python.exe -u \\
    tools/rig_pipeline/collect_rollout.py \\
    --task AniML-Deer-PolyArt-Walk-AMP-v0 \\
    --checkpoint logs/rsl_rl/animl_deer_polyart_walk/<run>/model_149.pt \\
    --num_envs 4 --steps 300 --out docs/polyart_policy_rollout.npz
"""
import argparse, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--task", default="AniML-Deer-PolyArt-Walk-AMP-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--out", default=str(REPO / "docs" / "polyart_policy_rollout.npz"))
os.environ.setdefault("ANIML_DEER_USD", str(REPO / "assets/deer_polyart_urdf/deer_polyart.usd"))

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args, extras = parser.parse_known_args()
sys.argv = [sys.argv[0]] + extras
app = AppLauncher(headless=True).app

import gymnasium as gym
import numpy as np
import torch

import training.tasks  # noqa: F401 — registers polyart task
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
from amp_rsl_rl.runners import AMPOnPolicyRunner
import importlib

env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
# Continuous render: disable EVERY termination except the timeout, and make the
# episode effectively infinite, so the clip shows raw sustained behaviour with no
# reset jumps. Disabling only `fell_over` was not enough -- `pelvis_too_low` still
# fired, teleporting the deer back to its origin mid-rollout. That silently
# corrupts both the video AND any net-displacement measurement taken from it
# (a reset reads as a huge instantaneous jump in root_pos).
for _term in list(vars(env_cfg.terminations).keys()):
    if _term != "time_out" and not _term.startswith("_"):
        setattr(env_cfg.terminations, _term, None)
env_cfg.episode_length_s = 10_000.0

task_spec = gym.spec(args.task)
mod_name, attr = task_spec.kwargs["amp_rsl_rl_cfg_entry_point"].split(":")
runner_cfg = dict(getattr(importlib.import_module(mod_name), attr))

env = gym.make(args.task, cfg=env_cfg, render_mode=None)
vec_env = RslRlVecEnvWrapper(env.unwrapped)
runner = AMPOnPolicyRunner(env=vec_env, train_cfg=runner_cfg, log_dir=None, device="cuda:0")
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device="cuda:0")
print(f"[rollout] loaded {args.checkpoint}", flush=True)

robot = env.unwrapped.scene["robot"]
joints_list = list(robot.data.joint_names)
jp_buf, rp_buf, rq_buf = [], [], []

# Per-body contact FORCES from PhysX. Inferring foot contact from FK hoof height is
# unreliable on this rig: the four hooves disagree about floor level by up to 5.8 cm
# (the 82 mm FK reconstruction error), so a height threshold mislabels stance/swing.
# The contact sensor is ground truth, and it is what the reward terms actually read.
contact_sensor = env.unwrapped.scene.sensors.get("contact_forces", None)
cf_buf = []
contact_bodies = list(contact_sensor.body_names) if contact_sensor is not None else []
if contact_sensor is None:
    print("[rollout] WARNING: no 'contact_forces' sensor; gait analysis will have to "
          "fall back to FK hoof heights", flush=True)

# RECORD THE COMMAND (added 2026-07-29). Without it a rollout is uninterpretable:
# every run so far was evaluated with a yaw command sampled in +/-0.5 rad/s (+/-29
# deg/s), and a deer that is correctly TURNING unloads its inside legs -- which reads
# in the analysis as "leg neglect". C3's rollout averaged +14 deg/s of yaw across all
# four envs and scored L/R load 0.14; without the command there is no way to tell that
# apart from a genuine pathology. Same failure shape as every other one on this
# project: the summary was fine, the CONDITION it was measured under was invisible.
cmd_buf = []
_cmd_mgr = getattr(env.unwrapped, "command_manager", None)

obs = vec_env.get_observations().to("cuda:0")
with torch.no_grad():
    for t in range(args.steps):
        actions = policy(obs)
        obs, _, _, _ = vec_env.step(actions.to(vec_env.device))
        obs = obs.to("cuda:0") if torch.is_tensor(obs) else obs
        jp_buf.append(robot.data.joint_pos.detach().cpu().numpy().copy())
        rp_buf.append(robot.data.root_pos_w.detach().cpu().numpy().copy())
        rq_buf.append(robot.data.root_quat_w.detach().cpu().numpy().copy())
        if contact_sensor is not None:
            # (num_envs, num_bodies) contact force magnitude in N
            f = torch.linalg.norm(contact_sensor.data.net_forces_w, dim=-1)
            cf_buf.append(f.detach().cpu().numpy().copy())
        if _cmd_mgr is not None:
            try:
                cmd_buf.append(_cmd_mgr.get_command("base_velocity")
                               .detach().cpu().numpy().copy())
            except Exception:
                _cmd_mgr = None   # log once by disabling, never spam the loop

# control rate for the render fps
ctrl_hz = round(1.0 / (env_cfg.sim.dt * env_cfg.decimation))
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
extra = {}
if cf_buf:
    extra["contact_forces"] = np.stack(cf_buf)          # (T, N, n_bodies) newtons
    extra["contact_bodies"] = np.array(contact_bodies)
if cmd_buf:
    extra["commands"] = np.stack(cmd_buf)               # (T, N, 3) vx, vy, yaw_rate
np.savez(
    args.out,
    joint_positions=np.stack(jp_buf),   # (T, N, 38)
    root_pos=np.stack(rp_buf),          # (T, N, 3)
    root_quat_wxyz=np.stack(rq_buf),    # (T, N, 4) wxyz
    joints_list=np.array(joints_list),
    fps=float(ctrl_hz),
    **extra,
)
print(f"[rollout] wrote {args.out}  frames={len(jp_buf)} envs={args.num_envs} fps={ctrl_hz}", flush=True)
print(f"[rollout] joints_list[:3]={joints_list[:3]}", flush=True)
env.close()
app.close()
