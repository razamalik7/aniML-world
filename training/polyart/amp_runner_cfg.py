"""amp-rsl-rl (2.2.3) runner config for the Tier-3 deer AMP walk.

A plain dict (AMPOnPolicyRunner needs a dict, not a dataclass). Written to the
**2.2.3** schema — verified against the installed package, matching the reminder
that we upgraded amp-rsl-rl mid-project:
  * ``obs_groups`` includes ``"actor"`` (rsl_rl >=4 renamed policy->actor).
  * ``actor`` / ``critic`` are ``MLPModel`` sections (rsl_rl 5.x branch); the
    legacy ``policy`` ActorCritic section is kept only for older rsl_rl. Actor's
    ``init_std`` lives in ``distribution_cfg`` (top-level ``init_noise_std`` is
    silently dropped by MLPModel).
  * native ``style_weight`` replaces the obsolete ``--task_reward_ratio`` monkey-
    patch: reward = (1-style_weight)*task + style_weight*style.
  * ``discriminator.loss_type = "BCEWithLogits"`` (not ...Loss).

Dataset: ``deer_polyart_walk`` (the relabeled 38-joint reference under
``assets/motions``). Override the directory with ``ANIML_AMP_DATA_PATH``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]


def _amp_data_path() -> Path:
    raw = os.environ.get("ANIML_AMP_DATA_PATH", "").strip()
    if raw and Path(raw).is_dir():
        return Path(raw).resolve()
    return (_REPO / "assets" / "motions").resolve()


def _amp_datasets() -> dict[str, float]:
    """Expert clips for the discriminator, as ``{name: sampling_weight}``.

    Override with ``ANIML_AMP_DATASETS="name:weight,name:weight"`` (weight optional,
    defaults to 1.0). Default stays the single DWalk take so existing runs are
    unchanged.

    WHY THIS IS A KNOB. Measured on r16p1 (tools/amp_status.py): ``amp_loss`` peaked at
    0.6449 -- near the ln(2)=0.6931 equilibrium where the discriminator genuinely
    cannot tell policy from expert -- then fell to **0.2985**, and style reward
    tracked it down 0.7507 -> 0.2963. That is textbook discriminator saturation, and
    it happened even under the LEGACY normaliser, which handicaps the discriminator by
    hiding policy outliers in inflated variance. The cause is quantity, not quality:
    the expert set is **33 frames / 1.1 s / one single stride**, against ~98,000 policy
    timesteps per iteration. We fixed the reference's quality three times (dead air,
    turn contamination, IK) and never its size.

    Available: deer_polyart_walk 33f | deer_polyart_walk_alltakes 96f (DWalk + DWalk R
    + DWalk L) | deer_polyart_trot 25f | deer_polyart_run 21f.

    CAUTION, two coupling constraints:
      * ``_alltakes`` contains TURNING takes, so the yaw command must be widened
        (``ANIML_DEER_CMD_ANG_VEL_Z``) or they are motion the command never asks for
        -- the r9 turn-contamination failure.
      * ``deer_polyart_run`` has a FLIGHT phase. Adding it while a hop pathology is
        live risks the discriminator legitimising exactly that.
    """
    raw = os.environ.get("ANIML_AMP_DATASETS", "").strip()
    if not raw:
        return {"deer_polyart_walk": 1.0}
    out: dict[str, float] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, w = part.partition(":")
        out[name.strip()] = float(w) if w.strip() else 1.0
    return out


DEER_POLYART_AMP_WALK_CFG: dict[str, Any] = {
    # -- runner
    "seed": 42,
    "num_steps_per_env": 24,
    "max_iterations": 3000,
    "save_interval": 100,
    "experiment_name": "animl_deer_polyart_walk",
    "run_name": "",
    "logger": "tensorboard",
    "resume": False,
    "load_run": ".*",
    "load_checkpoint": "model_.*.pt",
    "empirical_normalization": False,

    # rsl_rl obs-set routing. actor/critic read the env "policy" group; amp
    # reads the "amp" group the discriminator consumes. "actor" AND "policy"
    # both provided so either rsl_rl branch (>=4 uses "actor", 3.x "policy")
    # resolves.
    "obs_groups": {
        "actor": ["policy"],
        "policy": ["policy"],
        "critic": ["policy"],
        "amp": ["amp"],
    },

    # -- legacy ActorCritic (rsl_rl 3.x only; ignored on 5.x)
    "policy": {
        "class_name": "ActorCritic",
        "init_noise_std": 1.0,
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        "activation": "elu",
    },

    # -- rsl_rl >=4 MLPModel actor/critic (the branch we run on 5.0.1)
    "actor": {
        "class_name": "MLPModel",
        "hidden_dims": [512, 256, 128],
        "activation": "elu",
        "distribution_cfg": {
            "class_name": "GaussianDistribution",
            "init_std": 1.0,        # from-scratch exploration; AMP style reward
                                    # provides the gait prior. Lower to ~0.3 when
                                    # warm-starting.
            "std_type": "scalar",
        },
    },
    "critic": {
        "class_name": "MLPModel",
        "hidden_dims": [512, 256, 128],
        "activation": "elu",
    },

    # reward = (1-style_weight)*task + style_weight*style
    "style_weight": 0.5,

    # -- PPO + AMP
    "algorithm": {
        "class_name": "AMP_PPO",
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        # r3: 0.005->0.002. r4: ->0.001. Even at 0.002 noise_std kept GROWING
        # (1.42->1.49), keeping the gait noisy and risking a gait-destroying
        # balloon over the remaining iters. 0.001 forces it to decay (old rig
        # used 0.001 in v11/v12); watch for discriminator saturation at 0.
        # r10: 0.001 -> 0.003. The "watch for discriminator saturation" note below
        # came true. r9 measured Train/mean_style_reward COLLAPSING 0.7536 -> 0.2512
        # while Loss/amp_loss fell 0.6178 -> 0.2463 (discriminator winning), and
        # Train/mean_task_reward sat at ~-0.003 -- i.e. style was effectively the ONLY
        # learning signal and it decayed to a third of its strength. The result was a
        # policy that translated without ever forming a gait (measured: duty factor
        # 0.84-0.97 vs the reference's 0.27-0.67, and 3 of 4 legs barely lifting).
        # CLAUDE.md prescribes 0.002-0.005 to stop exactly this; 0.001 was below it.
        # Paired with a larger expert dataset (walk+trot+run), which is the other half
        # of the fix -- 55 expert frames were trivially memorisable.
        "entropy_coef": 0.003,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "learning_rate": 1.0e-3,
        "schedule": "adaptive",
        "gamma": 0.99,
        "lam": 0.95,
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
        "amp_replay_buffer_size": 100_000,
        "use_smooth_ratio_clipping": False,
    },

    # -- discriminator
    "discriminator": {
        "hidden_dims": [128, 128],
        "reward_scale": 1.0,
        "loss_type": "BCEWithLogits",
        "empirical_normalization": True,
    },

    # -- dataset
    "dataset": {
        "amp_data_path": _amp_data_path(),
        "datasets": _amp_datasets(),
        "slow_down_factor": 1,
    },
}
