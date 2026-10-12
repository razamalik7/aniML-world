"""Tier-3 (Poly Art / Malbers rig) training configs — the NEW generation.

Kept deliberately separate from the old ``deer_usd_*`` / URDF-v2 configs under
``training/envs`` and ``training/agents`` so the two generations never tangle.
Everything here targets the rebuilt, validated USD
``assets/deer_polyart_urdf/deer_polyart.usd`` (38 DOF) and its reference motion
``assets/motions/deer_polyart_walk.npy``.

Layout (mirrors ``tools/rig_pipeline/`` on the build side):
  * ``articulation.py``  — ArticulationCfg (actuator groups, spawn, contacts)
  * ``amp_env_cfg.py``   — AMP-augmented walk env (obs / rewards / RSI / terms)
  * ``amp_runner_cfg.py`` — amp-rsl-rl 2.2.3 runner dict
  * ``tasks.py``         — gym registration for the polyart tasks

``training/tasks/__init__.py`` imports ``training.polyart.tasks`` so the ids
register alongside the legacy tasks without mixing code.
"""
