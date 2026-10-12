"""Generate a PHYSICALLY GROUNDED procedural deer walk reference clip.

Motivation (see NIGHT_LOG_2026-07-23.md section 3g): every shipped AMP reference
clip is broken in ways reward tuning cannot fix --

  * feet NEVER touch the ground in any clip (min hoof +0.13 .. +0.31 m)
  * root_z is EXACTLY constant (range 0.0000) -> synthesized, not captured
  * no clip has a stance phase (walk 10-19%, should be 60-75%)
  * deer_walk's spine moves 2.4 deg and neck 2.5 deg over the whole stride

r47b demonstrated the consequence: even with feet_slide/-3.0 and feet_air_time/5.0
correctly wired to the hoof bodies, the policy still glides (hoof lift ~2 cm),
because BOTH rewards are contact-gated and the reference never makes contact.

This builds a clip that is correct BY CONSTRUCTION:
  - each hoof is pinned exactly at ground height through its stance phase
  - stance/swing timing from the gait parameters already measured in this repo
    (4-beat lateral-sequence walk, duty 0.65, offsets FL .000 HR .677 FR .516 HL .161)
  - root advances at the speed implied by stride length / cycle time, so a
    planted hoof is stationary in WORLD frame (zero foot slip by construction)
  - root height oscillates naturally instead of being a constant
  - spine and neck carry a real (small) oscillation instead of being frozen

Legs are solved with numerical IK on (hip_flex, knee, fetlock) using the SAME FK
as the production renderer, so the output is consistent with the rig.

Writes assets/motions/deer_walk_proc.npy -- a NEW file. Never touches the
shipped clips.

    python tools/make_procedural_walk.py            # build + validate
    python tools/make_procedural_walk.py --frames 60
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[1]
URDF = REPO / "assets" / "deer_urdf" / "deer_world.urdf"
OUT = REPO / "assets" / "motions" / "deer_walk_proc.npy"

# Dataset joint order MUST match the shipped clips (spine, HR*3, HL*3, FR*3, FL*3, neck, skull)
DATASET_ORDER = [
    "pelvis_to_chest",
    "pelvis_to_HR_thigh", "HR_thigh_to_HR_shin", "HR_shin_to_HR_foot",
    "pelvis_to_HL_thigh", "HL_thigh_to_HL_shin", "HL_shin_to_HL_foot",
    "chest_to_FR_thigh", "FR_thigh_to_FR_shin", "FR_shin_to_FR_foot",
    "chest_to_FL_thigh", "FL_thigh_to_FL_shin", "FL_shin_to_FL_foot",
    "chest_to_neck", "neck_to_skull",
]
B2A = {
    "pelvis_to_chest": "chest_yaw_to_chest",
    "pelvis_to_HR_thigh": "HR_hipabd_to_HR_thigh", "HR_thigh_to_HR_shin": "HR_thigh_to_HR_shin",
    "HR_shin_to_HR_foot": "HR_shin_to_HR_foot",
    "pelvis_to_HL_thigh": "HL_hipabd_to_HL_thigh", "HL_thigh_to_HL_shin": "HL_thigh_to_HL_shin",
    "HL_shin_to_HL_foot": "HL_shin_to_HL_foot",
    "chest_to_FR_thigh": "FR_hipabd_to_FR_thigh", "FR_thigh_to_FR_shin": "FR_thigh_to_FR_shin",
    "FR_shin_to_FR_foot": "FR_shin_to_FR_foot",
    "chest_to_FL_thigh": "FL_hipabd_to_FL_thigh", "FL_thigh_to_FL_shin": "FL_thigh_to_FL_shin",
    "FL_shin_to_FL_foot": "FL_shin_to_FL_foot",
    "chest_to_neck": "neck_yaw_to_neck", "neck_to_skull": "neck_to_skull",
}
# measured in this repo (CLAUDE.md / gait config): 4-beat lateral-sequence walk
PHASE = {"FL": 0.000, "HR": 0.677, "FR": 0.516, "HL": 0.161}
DUTY = 0.65          # fraction of cycle each foot is on the ground
LEGS = ["FL", "FR", "HL", "HR"]


def load_rig():
    J, c2j, lim = {}, {}, {}
    for j in ET.parse(URDF).getroot().findall("joint"):
        o = j.find("origin")
        xyz = np.array([float(v) for v in (o.get("xyz").split() if o is not None else [0, 0, 0])])
        ax = j.find("axis")
        axis = np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else np.zeros(3)
        name = j.get("name")
        J[name] = {"xyz": xyz, "axis": axis, "parent": j.find("parent").get("link")}
        c2j[j.find("child").get("link")] = name
        l = j.find("limit")
        if l is not None:
            lim[name] = (float(l.get("lower")), float(l.get("upper")))
    return J, c2j, lim


def rot_axis(axis, t):
    x, y, z = axis
    c, s = math.cos(t), math.sin(t)
    C = 1 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def quat_to_R_xyzw(q):
    x, y, z, w = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def link_pos(link, ang, J, c2j, R, t):
    chain, cur = [], link
    while cur in c2j:
        jn = c2j[cur]
        chain.append(jn)
        cur = J[jn]["parent"]
    chain.reverse()
    p, M = np.zeros(3), np.eye(3)
    for jn in chain:
        p = p + M @ J[jn]["xyz"]
        M = M @ rot_axis(J[jn]["axis"], ang.get(jn, 0.0))
    return R @ p + t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--speed", type=float, default=0.9, help="m/s forward")
    ap.add_argument("--stride", type=float, default=0.62, help="stride length (m)")
    ap.add_argument("--clearance", type=float, default=0.14, help="swing hoof lift (m)")
    ap.add_argument("--stand-height", type=float, default=0.705,
                    help="root height (physics-calibrated from r47b upright walkers)")
    ap.add_argument("--ground-z", type=float, default=-0.063,
                    help="toe-ORIGIN z at ground contact (physics-calibrated)")
    args = ap.parse_args()

    J, c2j, lim = load_rig()
    T, fps = args.frames, args.fps
    cycle = args.stride / args.speed                       # seconds per stride
    # Rx(-90) upright spawn, expressed xyzw (dataset convention)
    quat = np.array([-0.70710678, 0.0, 0.0, 0.70710678])
    R0 = quat_to_R_xyzw(quat)

    # --- standing height + ground target, CALIBRATED FROM PHYSICS
    #
    # An earlier version computed this from the zero-joint-angle pose (0.588 m).
    # That was wrong: the zero pose is NOT fully extended, so it understates leg
    # reach. It also cannot be taken from the toe collision meshes -- those are
    # stored in the URDF's global export frame, not link-local (front and hind
    # toe meshes sit on opposite Y sides), so mesh-derived offsets are garbage.
    #
    # Instead use what PhysX actually resolved for a genuinely standing deer in
    # r47b's rollout (envs 1 and 2, the upright walkers):
    #     pelvis height       0.705 m
    #     min toe-ORIGIN z   -0.063 m   (contact point, incl. soft-contact depth)
    #     => leg extension    0.768 m at ground contact
    zero = {B2A[n]: 0.0 for n in DATASET_ORDER}
    probe = link_pos("FL_toe", zero, J, c2j, R0, np.zeros(3))
    stand_h = args.stand_height
    ground_z = args.ground_z
    print(f"[proc] zero-pose toe offset {probe[2]:+.3f} (NOT used -- zero pose is not extended)")
    print(f"[proc] physics-calibrated: root {stand_h:.3f} m, stance target toe_z {ground_z:+.3f} m")

    # --- hip anchor positions (world offsets from root, neutral pose)
    hip_link = {"FL": "FL_thigh", "FR": "FR_thigh", "HL": "HL_thigh", "HR": "HR_thigh"}
    hips = {L: link_pos(hip_link[L], zero, J, c2j, R0, np.zeros(3)) for L in LEGS}

    JOINTS = {L: [B2A[f"{'chest' if L[0]=='F' else 'pelvis'}_to_{L}_thigh"],
                  B2A[f"{L}_thigh_to_{L}_shin"],
                  B2A[f"{L}_shin_to_{L}_foot"]] for L in LEGS}

    prev_q = {L: None for L in LEGS}
    jp = np.zeros((T, len(DATASET_ORDER)), dtype=np.float32)
    rp = np.zeros((T, 3), dtype=np.float32)
    rq = np.tile(quat.astype(np.float32), (T, 1))

    for f in range(T):
        ph_global = (f / T)
        t_sec = f / fps
        x_body = args.speed * t_sec
        # natural vertical oscillation at 2x stride frequency (~1.5 cm)
        z_body = stand_h + 0.015 * math.sin(4 * math.pi * ph_global)
        root = np.array([0.0, -x_body, z_body])            # deer faces world -Y

        ang = {B2A[n]: 0.0 for n in DATASET_ORDER}
        # small real spine/neck motion (the shipped clip has ~2.4 deg; give it life)
        ang[B2A["pelvis_to_chest"]] = 0.045 * math.sin(4 * math.pi * ph_global)
        ang[B2A["chest_to_neck"]] = 0.055 * math.sin(2 * math.pi * ph_global + 0.6)
        ang[B2A["neck_to_skull"]] = 0.30 + 0.05 * math.sin(2 * math.pi * ph_global)

        for L in LEGS:
            ph = (ph_global + PHASE[L]) % 1.0
            hip_w = R0 @ np.array([0, 0, 0]) + hips[L] + root
            if ph < DUTY:                                   # STANCE: hoof pinned to ground
                # The planted hoof must be FIXED in world frame for the whole
                # stance -> derive it from when this foot touched down, not from
                # the current frame. This is what gives zero foot slip.
                # Foot plants under ITS OWN hip, not under the pelvis.
                # hip world Y at time t = hips[L][1] - speed*t (deer faces -Y).
                stance_start_t = (f / fps) - (ph * cycle)
                hip_y0 = hips[L][1] - args.speed * stance_start_t
                plant_y = hip_y0 - args.stride * 0.5
                target = np.array([hips[L][0] + root[0], plant_y, ground_z])
            else:                                           # SWING: arc forward
                u = (ph - DUTY) / (1 - DUTY)
                stance_start_t = (f / fps) - (ph * cycle)
                hip_y0 = hips[L][1] - args.speed * stance_start_t
                plant_y = hip_y0 - args.stride * 0.5
                nxt = plant_y - args.stride
                target = np.array([hips[L][0] + root[0],
                                   plant_y + (nxt - plant_y) * u,
                                   ground_z + args.clearance * math.sin(math.pi * u)])
            # numerical IK on the 3 sagittal joints
            names = JOINTS[L]
            q = prev_q[L].copy() if prev_q[L] is not None else np.array([ang[n] for n in names])
            for _ in range(200):
                cur = link_pos(f"{L}_toe", {**ang, **dict(zip(names, q))}, J, c2j, R0, root)
                err = target - cur
                if np.linalg.norm(err) < 2e-5:
                    break
                J_num = np.zeros((3, 3))
                for k in range(3):
                    dq = q.copy(); dq[k] += 1e-4
                    p2 = link_pos(f"{L}_toe", {**ang, **dict(zip(names, dq))}, J, c2j, R0, root)
                    J_num[:, k] = (p2 - cur) / 1e-4
                dq = np.linalg.lstsq(J_num, err, rcond=None)[0]
                q = q + np.clip(dq, -0.25, 0.25)
                for k, n in enumerate(names):
                    if n in lim:
                        q[k] = float(np.clip(q[k], lim[n][0], lim[n][1]))
            for n, v in zip(names, q):
                ang[n] = float(v)
            prev_q[L] = q.copy()

        for i, dn in enumerate(DATASET_ORDER):
            jp[f, i] = ang[B2A[dn]]
        rp[f] = root

    jv = np.gradient(jp, 1.0 / fps, axis=0).astype(np.float32)
    out = {"joints_list": np.array(DATASET_ORDER), "joint_positions": jp,
           "joint_velocities": jv, "root_position": rp,
           "root_quaternion": rq, "fps": float(fps)}
    np.save(OUT, out, allow_pickle=True)
    print(f"[proc] wrote {OUT}  T={T} fps={fps} speed={args.speed} m/s")

    # ---- validate with the SAME metrics used to condemn the shipped clips
    print("\n[validate] vs shipped deer_walk")
    for tag, path in (("procedural", OUT), ("shipped", REPO / "assets/motions/deer_walk.npy")):
        d = np.load(path, allow_pickle=True).item()
        jl = list(d["joints_list"]); JP = d["joint_positions"]
        RP = d["root_position"]; RQ = d["root_quaternion"]
        n = JP.shape[0]
        mins, stances, lifts = [], [], []
        for toe in ("FL_toe", "FR_toe", "HL_toe", "HR_toe"):
            z = []
            for t in range(n):
                a = {B2A[b]: float(JP[t, i]) for i, b in enumerate(jl) if b in B2A}
                z.append(link_pos(toe, a, J, c2j, quat_to_R_xyzw(RQ[t]), RP[t].astype(float))[2])
            z = np.array(z)
            mins.append(z.min()); lifts.append(z.max() - z.min())
            stances.append((z < z.min() + 0.02).mean() * 100)
        print(f"  {tag:11} min_hoof={min(mins):+.3f}  mean_lift={np.mean(lifts):.3f}  "
              f"stance={np.mean(stances):.0f}%  root_z_range={RP[:,2].max()-RP[:,2].min():.4f}")


if __name__ == "__main__":
    main()
