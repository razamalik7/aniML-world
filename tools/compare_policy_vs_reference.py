"""Quantitative comparison: trained Stage 3 policy vs reference deer_walk anim.

Loads:
  - Reference: assets/motions/deer_walk.npy   (31 frames @ 30 fps, 15 DOF)
  - Policy:    logs/videos/play_loco_2026-05-05_07-17-20.npz
               (1000 frames @ 50 fps, 4 envs, 25 DOF)

Compares stride-relevant joints by name:
  4 knees     : *_thigh_to_*_shin
  4 cannons   : *_shin_to_*_foot
  head pitch  : neck_to_skull

For each: peak flex, avg |flex|, ROM, dominant frequency.
Plus root-frame forward velocity and yaw drift.

Output: prints summary table and writes
  docs/policy_vs_reference_walk_2026-05-05.md
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

REPO = Path(r"C:\Users\nars7\Projects\aniML-world\.claude\worktrees\jolly-clarke-e00da9")
REF_PATH = REPO / "assets" / "motions" / "deer_walk.npy"
POL_PATH = REPO / "logs" / "videos" / "play_loco_2026-05-05_23-10-29.npz"  # v4.6 model_7794
OUT_MD   = REPO / "docs" / "policy_v4p6_vs_reference_walk_2026-05-05.md"

KNEES   = ["FL_thigh_to_FL_shin", "FR_thigh_to_FR_shin",
           "HL_thigh_to_HL_shin", "HR_thigh_to_HR_shin"]
CANNONS = ["FL_shin_to_FL_foot", "FR_shin_to_FR_foot",
           "HL_shin_to_HL_foot", "HR_shin_to_HR_foot"]
HEAD    = "neck_to_skull"


def load_reference():
    d = np.load(REF_PATH, allow_pickle=True).item()
    joints = list(d["joints_list"])
    return {
        "joints": joints,
        "qpos": d["joint_positions"],            # (T, 15)
        "root_pos": d["root_position"],          # (T, 3)
        "root_quat": d["root_quaternion"],       # (T, 4)
        "fps": float(d["fps"]),
    }


def load_policy(env_idx: int = 0):
    d = np.load(POL_PATH, allow_pickle=True)
    joints = list(d["joints_list"])
    return {
        "joints": joints,
        "qpos": d["joint_positions"][:, env_idx, :],   # (T, 25)
        "root_pos": d["root_pos"][:, env_idx, :],
        "root_quat": d["root_quat_wxyz"][:, env_idx, :],
        "fps": float(d["fps"]),
    }


def quat_yaw_wxyz(q):
    """Return yaw (rad) from (T,4) wxyz quaternions."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def joint_traj(traj, name):
    if name not in traj["joints"]:
        return None
    idx = traj["joints"].index(name)
    return traj["qpos"][:, idx]


def stats_of(series, fps):
    a = np.asarray(series, dtype=np.float64)
    a_centered = a - a.mean()
    peak = float(np.max(np.abs(a)))
    rng  = float(a.max() - a.min())
    avg_abs = float(np.mean(np.abs(a)))
    rms = float(np.sqrt(np.mean(a_centered ** 2)))
    # dominant frequency via rfft on centered signal
    if len(a_centered) >= 8:
        spec = np.abs(np.fft.rfft(a_centered))
        freqs = np.fft.rfftfreq(len(a_centered), d=1.0 / fps)
        # ignore DC bin
        if len(spec) > 1:
            i = 1 + int(np.argmax(spec[1:]))
            dom_freq = float(freqs[i])
        else:
            dom_freq = 0.0
    else:
        dom_freq = 0.0
    return {"peak": peak, "rom": rng, "avg_abs": avg_abs, "rms": rms, "dom_hz": dom_freq}


def root_speed(traj, body_axis="−Y"):
    """Anatomical-forward speed (m/s).

    URDF convention: world -Y is the deer's facing direction at spawn.
    For the reference (which is straight-line authored along -Y), we just
    use ‖root_pos[t]-root_pos[0]‖ / dt.
    For the policy (which can yaw), project velocity onto the forward
    direction implied by current yaw. We use the simple body-frame |v_xy|
    as a robust estimate.
    """
    p = traj["root_pos"]
    fps = traj["fps"]
    dt = 1.0 / fps
    v = np.diff(p, axis=0) / dt              # (T-1, 3)
    speed_xy = np.linalg.norm(v[:, :2], axis=1)
    # Net displacement / total time
    net = float(np.linalg.norm(p[-1] - p[0]))
    total_t = (len(p) - 1) * dt
    net_speed = net / total_t if total_t > 0 else 0.0
    return {"mean_speed_xy": float(speed_xy.mean()),
            "net_displacement_speed": net_speed,
            "horizontal_path_len": float(np.sum(speed_xy) * dt)}


def yaw_drift(traj):
    yaw = quat_yaw_wxyz(traj["root_quat"])
    yaw_unwrapped = np.unwrap(yaw)
    total_t = (len(yaw) - 1) / traj["fps"]
    drift = float(yaw_unwrapped[-1] - yaw_unwrapped[0])
    return {"total_yaw_drift_rad": drift,
            "yaw_drift_deg_per_sec": float(np.degrees(drift) / total_t) if total_t > 0 else 0.0}


def fmt_rad(x):
    return f"{x:+.3f} rad ({np.degrees(x):+.1f}°)"


def main():
    ref = load_reference()
    pol = load_policy(env_idx=0)

    lines = []
    lines.append("# Stage 3 Policy vs Reference deer_walk — Stride Comparison")
    lines.append("")
    lines.append(f"- **Reference**: `{REF_PATH.name}` — {len(ref['qpos'])} frames @ {ref['fps']:.0f} fps "
                 f"= {len(ref['qpos'])/ref['fps']:.2f} s, 15 DOF")
    lines.append(f"- **Policy**:    `{POL_PATH.name}` — {len(pol['qpos'])} frames @ {pol['fps']:.0f} fps "
                 f"= {len(pol['qpos'])/pol['fps']:.2f} s, 25 DOF (env 0 of 4)")
    lines.append("")

    # ---- Root locomotion ----
    rs = root_speed(ref); ps = root_speed(pol)
    ry = yaw_drift(ref);  py = yaw_drift(pol)
    lines.append("## Root locomotion")
    lines.append("")
    lines.append("| Metric | Reference | Policy |")
    lines.append("|---|---|---|")
    lines.append(f"| Mean horizontal speed (m/s) | {rs['mean_speed_xy']:.3f} | {ps['mean_speed_xy']:.3f} |")
    lines.append(f"| Net-displacement speed (m/s) | {rs['net_displacement_speed']:.3f} | {ps['net_displacement_speed']:.3f} |")
    lines.append(f"| Total yaw drift | {fmt_rad(ry['total_yaw_drift_rad'])} | {fmt_rad(py['total_yaw_drift_rad'])} |")
    lines.append(f"| Yaw drift per sec | {ry['yaw_drift_deg_per_sec']:+.2f}°/s | {py['yaw_drift_deg_per_sec']:+.2f}°/s |")
    lines.append("")

    # ---- Knees ----
    lines.append("## Knee joints (the user-flagged 'knees not bending')")
    lines.append("")
    lines.append("| Joint | Source | peak\\|q\\| | ROM | RMS | dominant Hz |")
    lines.append("|---|---|---|---|---|---|")
    summary_knees = {"ref_peak": [], "pol_peak": [], "ref_rom": [], "pol_rom": []}
    for j in KNEES:
        for tag, t in [("ref", ref), ("pol", pol)]:
            s = joint_traj(t, j)
            if s is None:
                lines.append(f"| {j} | {tag} | _missing_ | | | |")
                continue
            st = stats_of(s, t["fps"])
            lines.append(f"| {j} | {tag} | {st['peak']:.3f} ({np.degrees(st['peak']):.1f}°) "
                         f"| {st['rom']:.3f} ({np.degrees(st['rom']):.1f}°) "
                         f"| {st['rms']:.3f} | {st['dom_hz']:.2f} |")
            summary_knees[f"{tag}_peak"].append(st["peak"])
            summary_knees[f"{tag}_rom"].append(st["rom"])
    lines.append("")
    avg_ref_peak = np.mean(summary_knees["ref_peak"])
    avg_pol_peak = np.mean(summary_knees["pol_peak"])
    avg_ref_rom  = np.mean(summary_knees["ref_rom"])
    avg_pol_rom  = np.mean(summary_knees["pol_rom"])
    lines.append(f"**Avg knee peak |q|:** ref {np.degrees(avg_ref_peak):.1f}°  vs  policy {np.degrees(avg_pol_peak):.1f}°  "
                 f"(policy is {(avg_pol_peak/avg_ref_peak*100):.0f}% of reference)")
    lines.append("")
    lines.append(f"**Avg knee ROM:**     ref {np.degrees(avg_ref_rom):.1f}°  vs  policy {np.degrees(avg_pol_rom):.1f}°  "
                 f"(policy is {(avg_pol_rom/avg_ref_rom*100):.0f}% of reference)")
    lines.append("")

    # ---- Cannons ----
    lines.append("## Cannon (shin->foot) joints")
    lines.append("")
    lines.append("| Joint | Source | peak\\|q\\| | ROM | RMS | dominant Hz |")
    lines.append("|---|---|---|---|---|---|")
    for j in CANNONS:
        for tag, t in [("ref", ref), ("pol", pol)]:
            s = joint_traj(t, j)
            if s is None:
                lines.append(f"| {j} | {tag} | _missing_ | | | |")
                continue
            st = stats_of(s, t["fps"])
            lines.append(f"| {j} | {tag} | {st['peak']:.3f} ({np.degrees(st['peak']):.1f}°) "
                         f"| {st['rom']:.3f} ({np.degrees(st['rom']):.1f}°) "
                         f"| {st['rms']:.3f} | {st['dom_hz']:.2f} |")
    lines.append("")

    # ---- Head pitch ----
    lines.append("## Head pitch (`neck_to_skull`)")
    lines.append("")
    lines.append("| Source | mean | peak\\|q\\| | ROM |")
    lines.append("|---|---|---|---|")
    for tag, t in [("ref", ref), ("pol", pol)]:
        s = joint_traj(t, HEAD)
        if s is None:
            continue
        lines.append(f"| {tag} | {fmt_rad(float(np.mean(s)))} | "
                     f"{fmt_rad(float(np.max(np.abs(s))))} | "
                     f"{fmt_rad(float(s.max()-s.min()))} |")
    lines.append("")

    # ---- Front vs hind activity asymmetry ----
    lines.append("## Front vs hind knee activity (RMS)")
    lines.append("")
    lines.append("| Source | Front-avg RMS | Hind-avg RMS | front/hind ratio |")
    lines.append("|---|---|---|---|")
    for tag, t in [("ref", ref), ("pol", pol)]:
        front = []
        hind  = []
        for j in KNEES:
            s = joint_traj(t, j)
            if s is None:
                continue
            r = stats_of(s, t["fps"])["rms"]
            (front if j.startswith("F") else hind).append(r)
        if front and hind:
            f = float(np.mean(front)); h = float(np.mean(hind))
            lines.append(f"| {tag} | {f:.4f} | {h:.4f} | {f/h:.2f} |")
    lines.append("")

    # ---- L/R asymmetry within hinds and within fronts ----
    lines.append("## L/R asymmetry (hind L vs hind R; front L vs front R)")
    lines.append("")
    lines.append("| Source | hindL/hindR RMS ratio | frontL/frontR RMS ratio |")
    lines.append("|---|---|---|")
    for tag, t in [("ref", ref), ("pol", pol)]:
        def rms_of(name):
            s = joint_traj(t, name)
            return stats_of(s, t["fps"])["rms"] if s is not None else None
        hl = rms_of("HL_thigh_to_HL_shin"); hr = rms_of("HR_thigh_to_HR_shin")
        fl = rms_of("FL_thigh_to_FL_shin"); fr = rms_of("FR_thigh_to_FR_shin")
        if hl and hr and fl and fr:
            lines.append(f"| {tag} | {hl/hr:.2f} | {fl/fr:.2f} |")
    lines.append("")

    # ---- One-line interpretation ----
    lines.append("## Bottom-line interpretation")
    lines.append("")
    lines.append(f"- **Knee bend depth**: policy reaches "
                 f"{(avg_pol_rom/avg_ref_rom*100):.0f}% of reference ROM. "
                 f"{'CONSISTENT WITH user observation that knees do not really bend.' if avg_pol_rom < 0.6*avg_ref_rom else 'Comparable to reference.'}")
    lines.append(f"- **Yaw drift**: policy drifts {py['yaw_drift_deg_per_sec']:+.1f}°/s vs reference "
                 f"{ry['yaw_drift_deg_per_sec']:+.1f}°/s. "
                 f"{'CONSISTENT WITH user observation of veering.' if abs(py['yaw_drift_deg_per_sec']) > 5 else 'Roughly straight.'}")
    lines.append("")

    out = "\n".join(lines)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(out, encoding="utf-8")
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        print(out)
    except UnicodeEncodeError:
        print(out.encode("ascii", "replace").decode("ascii"))
    print(f"\n[wrote] {OUT_MD}")


if __name__ == "__main__":
    main()
