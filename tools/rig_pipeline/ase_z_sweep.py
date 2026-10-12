"""Does the ASE latent actually DO anything? Sweep fixed z values and compare.

THE CORE ASE DIAGNOSTIC. Stage 1 only works if different z decode to visibly different
motion. If the policy learned to IGNORE z, every latent produces the same behaviour --
"latent collapse" -- and the whole two-stage architecture is pointless, because the
stage-2 high-level would have no dial to turn.

This matters for reading exp4's video: z is drawn RANDOMLY on every episode reset, so
that clip showed ONE arbitrary skill out of the sphere, not "what the policy does".
The front-legs-up propellor may be one bad corner of the latent rather than the policy
as a whole. Assigning each env a FIXED, DIFFERENT z and rendering them side by side is
the only way to tell those apart.

Each env gets its own constant z: env 0 = +e1, env 1 = -e1, env 2 = +e2, ... so the
sweep walks the sphere's principal axes. z is re-pinned every step, because the reset
EventTerm would otherwise resample it mid-rollout.

  python tools/rig_pipeline/ase_z_sweep.py --checkpoint <model.pt> --out docs/zsweep.npz
"""

import argparse, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--task", default="AniML-Deer-PolyArt-Walk-AMP-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=6)
parser.add_argument("--z_dim", type=int, default=8)
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

# --- assign each env a FIXED, DISTINCT latent ------------------------------
# Walk the sphere's principal axes: env0=+e1, env1=-e1, env2=+e2, env3=-e2, ...
# so neighbouring envs are exact opposites and any z-sensitivity shows up as a
# visible difference between them.
u = env.unwrapped
Z = torch.zeros(args.num_envs, args.z_dim, device=u.device)
for i in range(args.num_envs):
    Z[i, (i // 2) % args.z_dim] = 1.0 if i % 2 == 0 else -1.0
print("[zsweep] fixed latents per env:", flush=True)
for i in range(args.num_envs):
    print(f"    env {i}: z = {Z[i].cpu().numpy().round(2).tolist()}", flush=True)

def _pin_z():
    """Re-pin z every step: the reset EventTerm resamples it, which would silently
    turn a controlled sweep back into the random-z rollout we are trying to explain."""
    if getattr(u, "ase_z", None) is not None and u.ase_z.shape == Z.shape:
        u.ase_z.copy_(Z)

_pin_z()
obs = vec_env.get_observations().to("cuda:0")
with torch.no_grad():
    for t in range(args.steps):
        _pin_z()
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

# control rate for the render fps
ctrl_hz = round(1.0 / (env_cfg.sim.dt * env_cfg.decimation))
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
extra = {}
if cf_buf:
    extra["contact_forces"] = np.stack(cf_buf)          # (T, N, n_bodies) newtons
    extra["contact_bodies"] = np.array(contact_bodies)
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
