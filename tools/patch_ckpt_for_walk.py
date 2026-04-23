"""Patch a standing checkpoint for warm-starting walk training.

What this does:
  1. Loads the actor weights (which encode the standing policy).
  2. Resets distribution.std_param to a target value (default 0.4) so PPO
     has exploration capacity from the start of walk training.
  3. Resets critic_state_dict to fresh random weights — the critic's value
     estimates are completely wrong in the new (walk) reward landscape, so
     keeping them causes large value loss spikes and unstable LR schedules.
  4. Resets optimizer_state_dict for the same reason (stale momentum/variance
     from standing updates will push the walk policy in the wrong direction).
  5. Resets iter to 0 so curriculum ramps count from the beginning.

Usage:
    python tools/patch_ckpt_for_walk.py \
        --input  logs/rsl_rl/animl_deer_stand_v2/2026-04-13_22-09-04/model_199.pt \
        --output logs/rsl_rl/animl_deer_stand_v2/2026-04-13_22-09-04/model_199_walk_init.pt \
        --std    0.4
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch


def patch(input_path: str, output_path: str, target_std: float) -> None:
    src = Path(input_path)
    dst = Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(str(src), map_location="cpu", weights_only=False)

    actor = ckpt["actor_state_dict"]
    std_keys = [k for k in actor if "std" in k.lower()]
    if not std_keys:
        raise RuntimeError(f"No std param found in actor keys: {list(actor.keys())}")

    print(f"Checkpoint: {src.name}")
    print(f"Actor std keys: {std_keys}")
    for k in std_keys:
        old = actor[k].item() if actor[k].numel() == 1 else actor[k].mean().item()
        actor[k] = torch.full_like(actor[k], target_std)
        print(f"  {k}: {old:.5f} -> {target_std:.5f}")

    # Reset critic entirely — value landscape is completely different for walking
    critic = ckpt["critic_state_dict"]
    for k, v in critic.items():
        if "weight" in k:
            torch.nn.init.orthogonal_(v, gain=1.0)
        elif "bias" in k:
            torch.nn.init.zeros_(v)
    print("Critic weights re-initialized (orthogonal).")

    # Reset optimizer — stale momentum from standing updates will corrupt early walk gradients
    opt = ckpt["optimizer_state_dict"]
    opt["state"] = {}
    print("Optimizer state cleared.")

    # Reset iteration counter so curriculum ramps from step 0
    old_iter = ckpt.get("iter", "?")
    ckpt["iter"] = 0
    print(f"Iteration counter: {old_iter} -> 0")

    torch.save(ckpt, str(dst))
    print(f"\nSaved patched checkpoint to: {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--std",    type=float, default=0.4)
    args = parser.parse_args()
    patch(args.input, args.output, args.std)
