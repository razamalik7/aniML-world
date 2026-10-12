"""v4.7: Mirror-symmetry data augmentation for the deer locomotion policy.

Used by rsl_rl's PPO `symmetry_cfg["data_augmentation_func"]`. Doubles every
training batch with a mirrored (left<->right) copy, forcing the policy to
learn an L<->R-equivariant mapping from observations to actions.

OBSERVATION LAYOUT (95 dims, in order):
    0..2    base_lin_vel        (3, body-frame vector)
    3..5    base_ang_vel        (3, body-frame pseudovector)
    6..8    projected_gravity   (3, body-frame vector)
    9..11   velocity_commands   (3, anatomical: forward, lateral_right, yaw)
    12..36  joint_pos           (25, relative to default)
    37..61  joint_vel           (25)
    62..86  actions             (25, last action)
    87..94  gait_phase          (8, cos/sin per foot, order = FL HR FR HL)

ACTION LAYOUT (25 dims): joint position targets in the same order as joint_pos.

URDF axis convention (per CLAUDE.md):
    body +X = anatomical LEFT
    body +Y = anatomical DOWN
    body -Z = anatomical FORWARD

The L/R mirror plane is the YZ body plane (perpendicular to +X). Under that
reflection:
    vector  v: (vx, vy, vz) -> (-vx,  vy,  vz)   # vx component flips
    pseudo  w: (wx, wy, wz) -> ( wx, -wy, -wz)   # wx keeps, others flip
    grav g (vector): (gx, gy, gz) -> (-gx, gy, gz)

JOINT ORDER (from articulation, 25 DOF):
     0  pelvis_to_HL_hipabd
     1  pelvis_to_HR_hipabd
     2  pelvis_to_chest_yaw           (midline, yaw -> sign flip)
     3  HL_hipabd_to_HL_thigh
     4  HR_hipabd_to_HR_thigh
     5  chest_yaw_to_chest            (midline, pitch-like -> no sign flip)
     6  HL_thigh_to_HL_shin
     7  HR_thigh_to_HR_shin
     8  chest_to_FL_hipabd
     9  chest_to_FR_hipabd
    10  chest_to_neck_yaw             (midline, yaw -> sign flip)
    11  HL_shin_to_HL_foot
    12  HR_shin_to_HR_foot
    13  FL_hipabd_to_FL_thigh
    14  FR_hipabd_to_FR_thigh
    15  neck_yaw_to_neck              (midline, pitch-like -> no sign flip)
    16  HL_foot_to_HL_toe
    17  HR_foot_to_HR_toe
    18  FL_thigh_to_FL_shin
    19  FR_thigh_to_FR_shin
    20  neck_to_skull                 (midline, pitch -> no sign flip)
    21  FL_shin_to_FL_foot
    22  FR_shin_to_FR_foot
    23  FL_foot_to_FL_toe
    24  FR_foot_to_FR_toe
"""

from __future__ import annotations
import torch


# Joint permutation: out[i] = in[JOINT_PERM[i]]. Swaps every L<->R pair,
# leaves midline joints in place.
JOINT_PERM = [
    1, 0,                       # 0,1 hipabd hind
    2,                          # 2 pelvis_to_chest_yaw (midline)
    4, 3,                       # 3,4 thigh hind
    5,                          # 5 chest_yaw_to_chest (midline)
    7, 6,                       # 6,7 knee hind
    9, 8,                       # 8,9 hipabd front (shoulder abd)
    10,                         # 10 chest_to_neck_yaw (midline)
    12, 11,                     # 11,12 hock
    14, 13,                     # 13,14 thigh front
    15,                         # 15 neck_yaw_to_neck (midline)
    17, 16,                     # 16,17 pastern hind
    19, 18,                     # 18,19 elbow front
    20,                         # 20 neck_to_skull (midline)
    22, 21,                     # 21,22 carpus
    24, 23,                     # 23,24 pastern front
]
assert len(JOINT_PERM) == 25 and sorted(JOINT_PERM) == list(range(25))

# Joint sign flips applied AFTER the permutation. Most joints don't sign-flip
# under the mirror because the URDF was generated with mirrored joint axes
# (positive q on either side rotates outward in its respective half). Midline
# YAW joints DO sign-flip (turning right vs left).
JOINT_SIGNS = [
    +1, +1,        # 0,1 hipabd hind         (no flip - mirrored axes)
    -1,            # 2 pelvis_to_chest_yaw   (yaw, sign flip)
    +1, +1,        # 3,4 thigh hind
    +1,            # 5 chest_yaw_to_chest    (assumed pitch-like, no flip)
    +1, +1,        # 6,7 knee hind
    +1, +1,        # 8,9 hipabd front
    -1,            # 10 chest_to_neck_yaw    (yaw, sign flip)
    +1, +1,        # 11,12 hock
    +1, +1,        # 13,14 thigh front
    +1,            # 15 neck_yaw_to_neck     (assumed pitch-like, no flip)
    +1, +1,        # 16,17 pastern hind
    +1, +1,        # 18,19 elbow front
    +1,            # 20 neck_to_skull        (pitch, no flip)
    +1, +1,        # 21,22 carpus
    +1, +1,        # 23,24 pastern front
]
assert len(JOINT_SIGNS) == 25

# Body-frame vector / pseudovector signs (3-dim each)
LIN_VEL_SIGNS  = [-1, +1, +1]      # vector: vx flips
ANG_VEL_SIGNS  = [+1, -1, -1]      # pseudovector: wx keeps, others flip
GRAV_SIGNS     = [-1, +1, +1]      # vector: gx flips

# Velocity commands: anatomical (forward, lateral_right, yaw)
# - cmd_fwd: forward stays forward under mirror -> +1
# - cmd_lateral: right becomes left under mirror -> -1
# - cmd_yaw: clockwise becomes counter-clockwise under mirror -> -1
VEL_CMD_SIGNS = [+1, -1, -1]

# Gait-phase obs is 8 dims encoding (cos, sin) per foot, in foot order
# (FL, HR, FR, HL) per the GAIT_OFFSETS = (0.0, 0.0, 0.5, 0.5) tuple in env_cfg.
# Under L<->R mirror: FL <-> FR, HL <-> HR.
# Within each (cos,sin) pair, swapping feet alone is sufficient — no sign flip
# on cos or sin (both are scalar functions of phase).
# Foot index order in the 8-vector: FL(0,1), HR(2,3), FR(4,5), HL(6,7)
# After mirror swap (FL<->FR, HL<->HR):  FR(4,5), HL(6,7), FL(0,1), HR(2,3)
GAIT_PERM_LOCAL = [4, 5, 6, 7, 0, 1, 2, 3]


def _build_full_obs_perm_and_signs(device: torch.device):
    """Return (obs_perm, obs_signs) as torch tensors of shape (95,)."""
    obs_perm = []
    obs_signs = []

    # base_lin_vel (0..2)
    obs_perm.extend([0, 1, 2])
    obs_signs.extend(LIN_VEL_SIGNS)

    # base_ang_vel (3..5)
    obs_perm.extend([3, 4, 5])
    obs_signs.extend(ANG_VEL_SIGNS)

    # projected_gravity (6..8)
    obs_perm.extend([6, 7, 8])
    obs_signs.extend(GRAV_SIGNS)

    # velocity_commands (9..11)
    obs_perm.extend([9, 10, 11])
    obs_signs.extend(VEL_CMD_SIGNS)

    # joint_pos (12..36): permute via JOINT_PERM, apply JOINT_SIGNS
    obs_perm.extend([12 + j for j in JOINT_PERM])
    obs_signs.extend(JOINT_SIGNS)

    # joint_vel (37..61): same perm + signs
    obs_perm.extend([37 + j for j in JOINT_PERM])
    obs_signs.extend(JOINT_SIGNS)

    # actions (62..86): same perm + signs
    obs_perm.extend([62 + j for j in JOINT_PERM])
    obs_signs.extend(JOINT_SIGNS)

    # gait_phase (87..94): permute within block, no sign flips
    obs_perm.extend([87 + g for g in GAIT_PERM_LOCAL])
    obs_signs.extend([+1] * 8)

    assert len(obs_perm) == 95 and len(obs_signs) == 95
    return (
        torch.tensor(obs_perm,  device=device, dtype=torch.long),
        torch.tensor(obs_signs, device=device, dtype=torch.float32),
    )


def _build_action_perm_and_signs(device: torch.device):
    return (
        torch.tensor(JOINT_PERM,  device=device, dtype=torch.long),
        torch.tensor(JOINT_SIGNS, device=device, dtype=torch.float32),
    )


# Cache the perm tensors per-device. rsl_rl calls the augmentation function
# many times per training iter; rebuilding tensors each call is wasteful.
_PERM_CACHE: dict = {}


def _mirror_flat_obs(obs_tensor: torch.Tensor) -> torch.Tensor:
    """Apply mirror to a single (B, 95) flat obs tensor."""
    device = obs_tensor.device
    cache_key = ("perm", device)
    if cache_key not in _PERM_CACHE:
        obs_perm, obs_signs = _build_full_obs_perm_and_signs(device)
        act_perm, act_signs = _build_action_perm_and_signs(device)
        _PERM_CACHE[cache_key] = (obs_perm, obs_signs, act_perm, act_signs)
    obs_perm, obs_signs, _, _ = _PERM_CACHE[cache_key]
    return obs_tensor[:, obs_perm] * obs_signs.unsqueeze(0)


def _mirror_actions(actions: torch.Tensor) -> torch.Tensor:
    """Apply mirror to a (B, 25) action tensor."""
    device = actions.device
    cache_key = ("perm", device)
    if cache_key not in _PERM_CACHE:
        obs_perm, obs_signs = _build_full_obs_perm_and_signs(device)
        act_perm, act_signs = _build_action_perm_and_signs(device)
        _PERM_CACHE[cache_key] = (obs_perm, obs_signs, act_perm, act_signs)
    _, _, act_perm, act_signs = _PERM_CACHE[cache_key]
    return actions[:, act_perm] * act_signs.unsqueeze(0)


def mirror_obs_actions(env, obs, actions):
    """rsl_rl symmetry augmentation. Returns (obs_aug, actions_aug) where the
    batch dim is doubled: [original ; mirrored].

    obs: TensorDict with one key per observation group (e.g. "policy",
    optionally "critic"), each value of shape (B, 95). Or a plain (B, 95)
    tensor (handled for completeness / unit tests).

    actions: (B, 25) float tensor, or None.
    """
    # --- obs ---
    obs_aug = None
    if obs is not None:
        if isinstance(obs, torch.Tensor):
            # Plain tensor (e.g. unit test path)
            obs_mir = _mirror_flat_obs(obs)
            obs_aug = torch.cat([obs, obs_mir], dim=0)
        else:
            # TensorDict — assume each entry is (B, 95) flat
            from tensordict import TensorDict
            mirrored = {}
            for key in obs.keys():
                mirrored[key] = _mirror_flat_obs(obs[key])
            B = next(iter(obs.values())).shape[0]
            mirrored_td = TensorDict(mirrored, batch_size=[B], device=obs.device)
            obs_aug = torch.cat([obs, mirrored_td], dim=0)

    # --- actions ---
    actions_aug = None
    if actions is not None:
        actions_mir = _mirror_actions(actions)
        actions_aug = torch.cat([actions, actions_mir], dim=0)

    return obs_aug, actions_aug
