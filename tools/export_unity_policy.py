"""Export trained PolyArt Deer AMP/PPO checkpoints (D2 / D2b) for Unity 6 runtime inference.

Writes:
  - models/exported/deer_polyart_d2.onnx
  - models/exported/deer_polyart_d2b.onnx
  - models/exported/deer_polyart_spec.json
  - unity/aniML-world/Assets/_Project/Resources/Policies/deer_polyart_d2.bytes
  - unity/aniML-world/Assets/_Project/Resources/Policies/deer_polyart_d2b.bytes
"""
import json
import os
import shutil
import struct
import xml.etree.ElementTree as ET
import numpy as np
import torch


MAGIC = 0x4C4D4E41  # 'ANML'
VERSION = 1


def extract_fourier_cycle(data, env_idx, start_t, period, n_harmonics=6):
    t_slice = period * 3
    s = np.hstack([
        data["root_lin_vel"][start_t:start_t + t_slice, env_idx],
        data["root_ang_vel"][start_t:start_t + t_slice, env_idx],
        data["projected_gravity"][start_t:start_t + t_slice, env_idx],
        data["joint_pos"][start_t:start_t + t_slice, env_idx],
        data["joint_vel"][start_t:start_t + t_slice, env_idx],
    ])
    phase = 2.0 * np.pi * np.arange(t_slice) / float(period)
    basis = [np.ones_like(phase)]
    for k in range(1, n_harmonics + 1):
        basis.append(np.cos(k * phase))
        basis.append(np.sin(k * phase))
    b_mat = np.column_stack(basis).astype(np.float32)
    coeffs, _, _, _ = np.linalg.lstsq(b_mat, s, rcond=None)
    return coeffs.astype(np.float32)


def build_physics_operator(d2_rollout, d2b_rollout):
    x_list, y_list = [], []
    for data in [d2_rollout, d2b_rollout]:
        rpos = data["root_pos"]
        rlin = data["root_lin_vel"]
        rang = data["root_ang_vel"]
        grav = data["projected_gravity"]
        q = data["joint_pos"]
        qd = data["joint_vel"]
        act = data["actions"]
        t_steps, n_envs, _ = rpos.shape
        for e in range(n_envs):
            for t in range(t_steps - 1):
                if np.linalg.norm(rpos[t + 1, e, :2] - rpos[t, e, :2]) > 0.3:
                    continue
                s_t = np.concatenate([rlin[t, e], rang[t, e], grav[t, e], q[t, e], qd[t, e]])
                u_t = np.concatenate([s_t, act[t, e], [1.0]])
                s_next = np.concatenate([rlin[t + 1, e], rang[t + 1, e], grav[t + 1, e], q[t + 1, e], qd[t + 1, e]])
                x_list.append(u_t)
                y_list.append(s_next)

    x_mat = np.array(x_list, dtype=np.float32)
    y_mat = np.array(y_list, dtype=np.float32)
    reg = 1.0 * np.eye(x_mat.shape[1], dtype=np.float32)
    reg[-1, -1] = 0.0
    w_mat = np.linalg.solve(x_mat.T @ x_mat + reg, x_mat.T @ y_mat).astype(np.float32)
    return w_mat


def write_binary_bundle(out_path, ckpt_path, joint_names, action_ids, lo, hi, axes_isaac, w_mat, manifolds):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["actor_state_dict"]
    layers = [
        (sd["mlp.0.weight"].cpu().numpy().astype(np.float32), sd["mlp.0.bias"].cpu().numpy().astype(np.float32)),
        (sd["mlp.2.weight"].cpu().numpy().astype(np.float32), sd["mlp.2.bias"].cpu().numpy().astype(np.float32)),
        (sd["mlp.4.weight"].cpu().numpy().astype(np.float32), sd["mlp.4.bias"].cpu().numpy().astype(np.float32)),
        (sd["mlp.6.weight"].cpu().numpy().astype(np.float32), sd["mlp.6.bias"].cpu().numpy().astype(np.float32)),
    ]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<7i", MAGIC, VERSION, 122, 34, 38, 85, 6))
        f.write(np.array(action_ids, dtype=np.int32).tobytes())
        f.write(lo.astype(np.float32).tobytes())
        f.write(hi.astype(np.float32).tobytes())
        f.write(axes_isaac.astype(np.float32).tobytes())

        f.write(struct.pack("<i", len(layers)))
        for w, b in layers:
            out_dim, in_dim = w.shape
            f.write(struct.pack("<2i", in_dim, out_dim))
            f.write(w.tobytes())
            f.write(b.tobytes())

        phys_in, phys_out = w_mat.shape
        f.write(struct.pack("<2i", phys_in, phys_out))
        f.write(w_mat.tobytes())

        speeds = np.array([1.2, 1.8, 2.6], dtype=np.float32)
        freqs = np.array([1.16, 1.25, 1.35], dtype=np.float32)
        f.write(speeds.tobytes())
        f.write(freqs.tobytes())
        for m in manifolds:
            f.write(m.astype(np.float32).tobytes())

    print(f"Wrote Unity binary policy bundle: {out_path} ({os.path.getsize(out_path):,} bytes)")


def main():
    repo_root = r"c:\Users\nars7\Projects\aniML-world"
    d2_rollout_path = r"C:\Users\nars7\.gemini\antigravity\brain\41971cea-1b98-43de-b409-d1fd3b6adbfe\scratch\d2_multi_rollout.npz"
    d2b_rollout_path = r"C:\Users\nars7\.gemini\antigravity\brain\41971cea-1b98-43de-b409-d1fd3b6adbfe\scratch\d2b_multi_rollout.npz"

    d2_rollout = np.load(d2_rollout_path)
    d2b_rollout = np.load(d2b_rollout_path)
    joint_names = [str(x) for x in d2_rollout["joint_names"]]
    action_joint_names = [str(x) for x in d2_rollout["action_joint_names"]]
    action_ids = [int(x) for x in d2_rollout["action_joint_ids"]]

    urdf_path = os.path.join(repo_root, "assets", "deer_polyart_urdf", "deer_polyart.urdf")
    tree = ET.parse(urdf_path)
    lo = np.zeros(38, dtype=np.float32)
    hi = np.zeros(38, dtype=np.float32)
    axes_isaac = np.zeros((38, 3), dtype=np.float32)
    joint_meta = []

    urdf_map = {}
    for j in tree.getroot().findall("joint"):
        name = j.attrib["name"]
        urdf_map[name] = {
            "parent": j.find("parent").attrib["link"],
            "child": j.find("child").attrib["link"],
            "lo": float(j.find("limit").attrib["lower"]),
            "hi": float(j.find("limit").attrib["upper"]),
            "axis_isaac": [float(x) for x in j.find("axis").attrib["xyz"].split()],
        }

    for idx, name in enumerate(joint_names):
        info = urdf_map[name]
        lo[idx] = info["lo"]
        hi[idx] = info["hi"]
        axes_isaac[idx] = info["axis_isaac"]
        ax_i = info["axis_isaac"]
        ax_u = [ax_i[1], -ax_i[2], -ax_i[0]]
        child = info["child"]
        unity_bone = (child[:-5] if child.endswith("_dof0") else child).replace("_", " ")
        dof_slot = 0 if child.endswith("_dof0") or not any(
            k.endswith("_dof0") and k[:-5] == child for k in [u["child"] for u in urdf_map.values()]
        ) else 1
        joint_meta.append({
            "index": idx,
            "name": name,
            "unity_bone": unity_bone,
            "dof_slot": dof_slot,
            "is_active": idx in action_ids,
            "action_index": action_ids.index(idx) if idx in action_ids else -1,
            "lower": round(float(lo[idx]), 4),
            "upper": round(float(hi[idx]), 4),
            "axis_isaac": [round(float(x), 6) for x in ax_i],
            "axis_unity_root": [round(float(x), 6) for x in ax_u],
        })

    w_mat = build_physics_operator(d2_rollout, d2b_rollout)
    manifolds_d2 = [
        extract_fourier_cycle(d2_rollout, 0, 40, 43, 6),
        extract_fourier_cycle(d2_rollout, 1, 40, 40, 6),
        extract_fourier_cycle(d2_rollout, 2, 40, 37, 6),
    ]
    manifolds_d2b = [
        extract_fourier_cycle(d2b_rollout, 0, 40, 43, 6),
        extract_fourier_cycle(d2b_rollout, 1, 40, 40, 6),
        extract_fourier_cycle(d2b_rollout, 2, 40, 37, 6),
    ]

    export_dir = os.path.join(repo_root, "models", "exported")
    os.makedirs(export_dir, exist_ok=True)

    d2_dir = r"C:\Users\nars7\Projects\aniML-world\.claude\worktrees\jolly-clarke-e00da9\logs\rsl_rl\animl_deer_polyart_walk\2026-07-29_00-33-54"
    d2b_dir = r"C:\Users\nars7\Projects\aniML-world\.claude\worktrees\jolly-clarke-e00da9\logs\rsl_rl\animl_deer_polyart_walk\2026-08-09_10-48-41"

    shutil.copy2(os.path.join(d2_dir, "policy_2699.onnx"), os.path.join(export_dir, "deer_polyart_d2.onnx"))
    shutil.copy2(os.path.join(d2b_dir, "policy_3898.onnx"), os.path.join(export_dir, "deer_polyart_d2b.onnx"))

    spec = {
        "species": "deer_polyart",
        "control_hz": 50.0,
        "sim_dt": 0.005,
        "decimation": 4,
        "obs_dim": 122,
        "action_dim": 34,
        "joint_count": 38,
        "observation_slices": {
            "base_lin_vel_b": [0, 3],
            "base_ang_vel_b": [3, 6],
            "projected_gravity_b": [6, 9],
            "velocity_commands": [9, 12],
            "joint_pos": [12, 50],
            "joint_vel": [50, 88],
            "last_action": [88, 122],
        },
        "joint_names_38": joint_names,
        "action_joint_names_34": action_joint_names,
        "action_joint_ids_34": action_ids,
        "joints": joint_meta,
    }
    spec_path = os.path.join(export_dir, "deer_polyart_spec.json")
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)
    print(f"Wrote spec: {spec_path}")

    unity_pol_dir = os.path.join(repo_root, "unity", "aniML-world", "Assets", "_Project", "Resources", "Policies")
    write_binary_bundle(
        os.path.join(unity_pol_dir, "deer_polyart_d2.bytes"),
        os.path.join(d2_dir, "model_2699.pt"),
        joint_names, action_ids, lo, hi, axes_isaac, w_mat, manifolds_d2,
    )
    write_binary_bundle(
        os.path.join(unity_pol_dir, "deer_polyart_d2b.bytes"),
        os.path.join(d2b_dir, "model_3898.pt"),
        joint_names, action_ids, lo, hi, axes_isaac, w_mat, manifolds_d2b,
    )


if __name__ == "__main__":
    main()
