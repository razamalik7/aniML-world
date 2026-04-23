"""Locomotion reward weights (documentation mirror of ARCHITECTURE.md §4)."""

# Weights for reference when adding custom RewTermCfg entries in env configs.
VELOCITY_TRACKING_LIN = 1.0
VELOCITY_TRACKING_ANG = 0.5
BODY_HEIGHT = -2.0
ORIENTATION_PENALTY = -1.0
ENERGY_PENALTY = -0.001
FEET_AIR_TIME = 0.5
ACTION_RATE = -0.01
FEET_STUMBLE = -1.0
