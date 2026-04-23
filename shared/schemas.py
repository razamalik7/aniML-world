"""
Tensor layout contract: Isaac Lab (train) ↔ Unity Sentis (deploy).

Regenerate JSON for C# consumers:
  python shared/schemas.py --export
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# --- Dimensions (locomotion phase) ---
JOINT_POS_DIM = 12
JOINT_VEL_DIM = 12
BODY_ORIENTATION_DIM = 4
BODY_ANGULAR_VEL_DIM = 3
GRAVITY_PROJECTION_DIM = 3
VELOCITY_COMMAND_DIM = 3
BODY_LINEAR_VEL_DIM = 2
LAST_ACTION_DIM = 9
HEIGHT_SCAN_DIM = 100
CONTEXT_TOKEN_DIM = 11

LOC_OBS_DIM = (
    JOINT_POS_DIM
    + JOINT_VEL_DIM
    + BODY_ORIENTATION_DIM
    + BODY_ANGULAR_VEL_DIM
    + GRAVITY_PROJECTION_DIM
    + VELOCITY_COMMAND_DIM
    + BODY_LINEAR_VEL_DIM
    + LAST_ACTION_DIM
    + HEIGHT_SCAN_DIM
)
ECO_OBS_DIM = LOC_OBS_DIM + CONTEXT_TOKEN_DIM
ACTION_DIM = 12

CONTEXT_TOKEN_IDS: dict[str, int] = {
    "CLEAR_SKY": 0,
    "LIGHT_RAIN": 1,
    "HEAVY_RAIN": 2,
    "THUNDERSTORM": 3,
    "METEOR_INCOMING": 4,
    "METEOR_IMPACT": 5,
    "NIGHT": 6,
    "DAWN": 7,
    "DUSK": 8,
    "DROUGHT": 9,
    "WILDFIRE": 10,
}

OBS_SLICES_LOC: dict[str, tuple[int, int]] = {
    "joint_positions": (0, 12),
    "joint_velocities": (12, 24),
    "body_orientation": (24, 28),
    "body_angular_velocity": (28, 31),
    "gravity_projection": (31, 34),
    "velocity_command": (34, 37),
    "body_linear_velocity": (37, 39),
    "last_action": (39, 48),
    "height_scan": (48, 148),
    "context_tokens": (148, 159),
}

ACTION_SLICES: dict[str, tuple[int, int]] = {
    "joint_pos_targets": (0, 12),
}


def export_json(path: Path | None = None) -> Path:
    """Write shared/schemas.json next to this file unless path is given."""
    out = path or Path(__file__).resolve().parent / "schemas.json"
    payload = {
        "locomotion_obs_dim": LOC_OBS_DIM,
        "ecosystem_obs_dim": ECO_OBS_DIM,
        "action_dim": ACTION_DIM,
        "context_token_dim": CONTEXT_TOKEN_DIM,
        "context_token_ids": CONTEXT_TOKEN_IDS,
        "observation_slices_locomotion": {
            k: {"start": v[0], "end": v[1], "dim": v[1] - v[0]}
            for k, v in OBS_SLICES_LOC.items()
            if v[1] <= LOC_OBS_DIM or k == "context_tokens"
        },
        "action_slices": {
            k: {"start": v[0], "end": v[1], "dim": v[1] - v[0]} for k, v in ACTION_SLICES.items()
        },
        "joint_order_legs": ["front_left", "front_right", "rear_left", "rear_right"],
        "joint_order_per_leg": ["hip_yaw", "hip_pitch", "knee"],
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--export", action="store_true", help="Write schemas.json")
    args = p.parse_args()
    if args.export:
        out = export_json()
        print(f"Wrote {out}")
    else:
        p.print_help()


if __name__ == "__main__":
    main()
