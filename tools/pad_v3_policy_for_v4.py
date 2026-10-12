"""Pad a v3-trained rsl_rl checkpoint (21-action / 83-obs policy) into a
v4-compatible checkpoint (25-action / 95-obs policy) by inserting zeros for
the 4 new pastern joints at their correct positions in BOTH:
  - v4 joint ordering (action space dim 21 -> 25)
  - v4 observation layout (joint_pos / joint_vel / last_action grew by 4 each)

Joint index mapping (verified via tools/dump_joint_order.py):
  v4 idx 0..15 = v3 idx 0..15        (same joints)
  v4 idx 16    = NEW HL_foot_to_HL_toe
  v4 idx 17    = NEW HR_foot_to_HR_toe
  v4 idx 18..22 = v3 idx 16..20
  v4 idx 23    = NEW FL_foot_to_FL_toe
  v4 idx 24    = NEW FR_foot_to_FR_toe

Observation layout (per ANYmal-C / Walk-These-Ways convention used by parent):
  obs[0:12]  base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3) + velocity_cmd(3)
  obs[12:N1] joint_pos      (N joints, where N = 21 for v3, 25 for v4)
  obs[N1:N2] joint_vel      (N joints)
  obs[N2:N3] last_action    (N joints)
  obs[N3:end] gait_phase    (8)

For each of the 3 joint-indexed obs sections, we insert 4 zero columns at the
v4 indices [16, 17, 23, 24] within that section.

What's padded:
  actor.mlp.0.weight   (512, 83) -> (512, 95)   new columns = 0 (zero on new obs)
  actor.mlp.6.weight   (21, 128) -> (25, 128)   new rows = 0 (zero action mean)
  actor.mlp.6.bias     (21,) -> (25,)
  actor.distribution.std_param  (21,) -> (25,)  new entries = init_noise_std
  critic.mlp.0.weight  (512, 83) -> (512, 95)   new columns = 0 (same obs layout)

Optimizer state dropped (Adam rebuilds momentum from zero).

Usage:
  python tools/pad_v3_policy_for_v4.py <v3_model_1999.pt> <output.pt>
"""
from __future__ import annotations
import sys
import torch

V3_TO_V4_INDEX = {
    # v3_idx: v4_idx
    **{i: i for i in range(0, 16)},        # first 16 unchanged
    16: 18, 17: 19, 18: 20, 19: 21, 20: 22,  # shifted by +2 due to HL_toe/HR_toe insertion at 16-17
}
# v4 indices 16, 17, 23, 24 are NEW (no v3 source).
NEW_V4_INDICES = [16, 17, 23, 24]
V4_DIM = 25
V3_DIM = 21

# Initial noise std for the new pastern actions (same as PPO config init_noise_std=0.5).
INIT_STD_FOR_NEW_PASTERNS = 0.5


def pad_action_dim(v3_tensor: torch.Tensor, axis_pad: int, new_value: float):
    """Pad along dim 0 from V3_DIM to V4_DIM, mapping v3 indices to v4 per V3_TO_V4_INDEX.
    New v4 indices get *new_value*."""
    assert v3_tensor.shape[0] == V3_DIM, f"expected first dim {V3_DIM}, got {tuple(v3_tensor.shape)}"
    new_shape = (V4_DIM,) + tuple(v3_tensor.shape[1:])
    out = torch.full(new_shape, new_value, dtype=v3_tensor.dtype)
    for v3_i, v4_i in V3_TO_V4_INDEX.items():
        out[v4_i] = v3_tensor[v3_i]
    return out


# Observation layout
# v3: 12 (base/cmd) + 21 (joint_pos) + 21 (joint_vel) + 21 (last_action) + 8 (gait) = 83
# v4: 12 (base/cmd) + 25 (joint_pos) + 25 (joint_vel) + 25 (last_action) + 8 (gait) = 95
OBS_HEAD_DIM = 12          # base_lin_vel + ang_vel + gravity + cmd  (unchanged)
OBS_GAIT_DIM = 8           # gait phase tail (unchanged)
V3_OBS_DIM = 83
V4_OBS_DIM = 95


def build_obs_v3_to_v4_index():
    """Return a dict mapping v3 obs index -> v4 obs index.
    Indices not present in the dict are NEW in v4 (initialized to zero).
    """
    mapping = {}
    # Head section [0:12] — identical
    for i in range(OBS_HEAD_DIM):
        mapping[i] = i
    # joint_pos: v3[12:33] -> v4[12:37]; pasterns inserted at v4 indices 16,17,23,24
    for v3_local, v4_local in V3_TO_V4_INDEX.items():
        mapping[OBS_HEAD_DIM + v3_local] = OBS_HEAD_DIM + v4_local
    # joint_vel: v3[33:54] -> v4[37:62]
    JV_V3 = OBS_HEAD_DIM + V3_DIM
    JV_V4 = OBS_HEAD_DIM + V4_DIM
    for v3_local, v4_local in V3_TO_V4_INDEX.items():
        mapping[JV_V3 + v3_local] = JV_V4 + v4_local
    # last_action: v3[54:75] -> v4[62:87]
    LA_V3 = OBS_HEAD_DIM + 2 * V3_DIM
    LA_V4 = OBS_HEAD_DIM + 2 * V4_DIM
    for v3_local, v4_local in V3_TO_V4_INDEX.items():
        mapping[LA_V3 + v3_local] = LA_V4 + v4_local
    # gait_phase tail [last 8] — identical, comes after last_action
    GP_V3 = OBS_HEAD_DIM + 3 * V3_DIM
    GP_V4 = OBS_HEAD_DIM + 3 * V4_DIM
    for i in range(OBS_GAIT_DIM):
        mapping[GP_V3 + i] = GP_V4 + i
    return mapping


def pad_input_layer_columns(weight_v3: torch.Tensor) -> torch.Tensor:
    """Pad a Linear weight (out, in_v3) -> (out, in_v4) along dim 1.
    New columns initialized to zero — meaning the policy ignores the new
    observation values until learning shifts them."""
    assert weight_v3.shape[1] == V3_OBS_DIM, f"expected in_dim {V3_OBS_DIM}, got {tuple(weight_v3.shape)}"
    obs_map = build_obs_v3_to_v4_index()
    out = torch.zeros((weight_v3.shape[0], V4_OBS_DIM), dtype=weight_v3.dtype)
    for v3_i, v4_i in obs_map.items():
        out[:, v4_i] = weight_v3[:, v3_i]
    return out


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    src = sys.argv[1]
    dst = sys.argv[2]
    print(f"[pad] loading {src}")
    ckpt = torch.load(src, map_location="cpu", weights_only=False)

    actor = ckpt["actor_state_dict"]
    critic = ckpt.get("critic_state_dict", {})
    print("[pad] actor before:")
    for k in actor:
        if hasattr(actor[k], "shape"):
            print(f"  {k}: {tuple(actor[k].shape)}")

    # ---- Output side (action dim 21 -> 25) ----
    actor["distribution.std_param"] = pad_action_dim(
        actor["distribution.std_param"], axis_pad=0, new_value=INIT_STD_FOR_NEW_PASTERNS
    )
    actor["mlp.6.bias"] = pad_action_dim(actor["mlp.6.bias"], axis_pad=0, new_value=0.0)
    w_out = actor["mlp.6.weight"]
    new_w_out = torch.zeros((V4_DIM, w_out.shape[1]), dtype=w_out.dtype)
    for v3_i, v4_i in V3_TO_V4_INDEX.items():
        new_w_out[v4_i] = w_out[v3_i]
    actor["mlp.6.weight"] = new_w_out

    # ---- Input side (obs dim 83 -> 95) ----
    actor["mlp.0.weight"] = pad_input_layer_columns(actor["mlp.0.weight"])
    if critic and "mlp.0.weight" in critic:
        critic["mlp.0.weight"] = pad_input_layer_columns(critic["mlp.0.weight"])

    print("[pad] actor after:")
    for k in actor:
        if hasattr(actor[k], "shape"):
            print(f"  {k}: {tuple(actor[k].shape)}")
    print("[pad] critic after:")
    for k in critic:
        if hasattr(critic[k], "shape"):
            print(f"  {k}: {tuple(critic[k].shape)}")

    # Drop optimizer state — Adam rebuilds momentum from zero
    if "optimizer_state_dict" in ckpt:
        ckpt["optimizer_state_dict"] = {"state": {}, "param_groups": ckpt["optimizer_state_dict"].get("param_groups", [])}

    ckpt["iter"] = 0

    print(f"[pad] saving to {dst}")
    torch.save(ckpt, dst)
    print(f"[pad] done.")
    print(f"      Action dim {V3_DIM} -> {V4_DIM} (zeros at v4 indices {NEW_V4_INDICES})")
    print(f"      Obs dim   {V3_OBS_DIM} -> {V4_OBS_DIM} (12 new zero cols for new joint_pos/vel/action obs)")
    print(f"      New pastern std_param entries = {INIT_STD_FOR_NEW_PASTERNS}")


if __name__ == "__main__":
    main()
