"""
Export actor policy to ONNX (opset 17) for Unity Sentis.

Run with Isaac Sim Kit Python (has torch), from repo root on PYTHONPATH:

  isaaclab.bat -p training/scripts/export_onnx.py --out models/exported/deer_locomotion_v1.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)

    try:
        import torch
    except ImportError as e:
        raise SystemExit("PyTorch not found; run this script with isaaclab.bat -p") from e

    from shared.schemas import ACTION_DIM, LOC_OBS_DIM
    from models.networks import ActorExportWrapper, build_actor_critic

    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("models/exported/deer_locomotion_v1.onnx"))
    p.add_argument("--obs-dim", type=int, default=LOC_OBS_DIM, help="Training observation size")
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()

    ac = build_actor_critic(args.obs_dim, ACTION_DIM)
    wrapper = ActorExportWrapper(ac)
    wrapper.eval()
    dummy = torch.randn(1, args.obs_dim)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        str(args.out),
        input_names=["observations"],
        output_names=["action_mean"],
        dynamic_axes={"observations": {0: "batch"}, "action_mean": {0: "batch"}},
        opset_version=args.opset,
    )
    print(f"Wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
