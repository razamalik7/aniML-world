import sys; sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from pathlib import Path

ROOT = Path(r"C:\Users\nars7\Projects\aniML-world\.claude\worktrees\jolly-clarke-e00da9\assets\motions")
for name in ["deer_walk.npy", "deer_run.npy", "deer_idle.npy"]:
    p = ROOT / name
    d = np.load(p, allow_pickle=True).item()
    j = list(d["joints_list"])
    qp = d["joint_positions"]
    rp = d["root_position"]
    fps = float(d["fps"])
    secs = qp.shape[0] / fps
    print(f"{name}: {qp.shape[0]} frames @ {fps:.0f}fps = {secs:.2f}s")
    for jn in ["FL_thigh_to_FL_shin","FR_thigh_to_FR_shin","HL_thigh_to_HL_shin","HR_thigh_to_HR_shin","neck_to_skull"]:
        if jn in j:
            i = j.index(jn)
            rom = float(qp[:, i].max() - qp[:, i].min())
            mean = float(qp[:, i].mean())
            print(f"   {jn}: ROM={np.degrees(rom):.1f}deg  mean={np.degrees(mean):.1f}deg")
    speed = float(np.linalg.norm(rp[-1] - rp[0]) / secs)
    print(f"   net forward speed: {speed:.2f} m/s")
    print()
