import sys; sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

d = np.load(r"C:\Users\nars7\Projects\aniML-world\.claude\worktrees\jolly-clarke-e00da9\logs\videos\play_loco_2026-05-05_07-17-20.npz", allow_pickle=True)
joints = list(d["joints_list"])
qpos = d["joint_positions"]
rpos = d["root_pos"]
rquat = d["root_quat_wxyz"]
fps = float(d["fps"])

def yaw(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

fl = joints.index("FL_thigh_to_FL_shin")
fr = joints.index("FR_thigh_to_FR_shin")
hl = joints.index("HL_thigh_to_HL_shin")
hr = joints.index("HR_thigh_to_HR_shin")
hd = joints.index("neck_to_skull")

hdr = ["env", "net_m", "pathlen_m", "yaw_drift_deg", "mean_speed_mps",
       "FL_knee_ROM_deg", "FR_knee_ROM_deg", "HL_knee_ROM_deg", "HR_knee_ROM_deg",
       "head_mean_deg"]
print(" | ".join(f"{h:>15}" for h in hdr))
for e in range(qpos.shape[1]):
    p = rpos[:, e, :]
    q = rquat[:, e, :]
    v = np.diff(p, axis=0) * fps
    speed_xy = np.linalg.norm(v[:, :2], axis=1)
    net = float(np.linalg.norm(p[-1, :2] - p[0, :2]))
    pathlen = float(np.sum(speed_xy) / fps)
    yw = np.unwrap(yaw(q))
    drift_deg = float(np.degrees(yw[-1] - yw[0]))
    row = [e,
           round(net, 3), round(pathlen, 3), round(drift_deg, 2),
           round(float(speed_xy.mean()), 3),
           round(float(np.degrees(qpos[:, e, fl].max() - qpos[:, e, fl].min())), 1),
           round(float(np.degrees(qpos[:, e, fr].max() - qpos[:, e, fr].min())), 1),
           round(float(np.degrees(qpos[:, e, hl].max() - qpos[:, e, hl].min())), 1),
           round(float(np.degrees(qpos[:, e, hr].max() - qpos[:, e, hr].min())), 1),
           round(float(np.degrees(np.mean(qpos[:, e, hd]))), 2)]
    print(" | ".join(f"{x:>15}" for x in row))

# Path efficiency (net / pathlen) per env
print()
print("Path efficiency (net / pathlen): low = walking in circles / station-keeping")
for e in range(qpos.shape[1]):
    p = rpos[:, e, :]
    v = np.diff(p, axis=0) * fps
    speed_xy = np.linalg.norm(v[:, :2], axis=1)
    pathlen = float(np.sum(speed_xy) / fps)
    net = float(np.linalg.norm(p[-1, :2] - p[0, :2]))
    eff = net / pathlen if pathlen > 0 else 0.0
    print(f"  env {e}: net={net:.2f} m, path={pathlen:.2f} m, efficiency={eff:.2%}")
