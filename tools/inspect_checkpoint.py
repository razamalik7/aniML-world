"""Dump the structure of a rsl_rl checkpoint so we can plan weight padding."""
import sys, torch, json
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: inspect_checkpoint.py <model.pt>")
    sys.exit(1)

ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(f"Top-level keys: {list(ckpt.keys())}")
for k in ckpt:
    v = ckpt[k]
    if isinstance(v, dict):
        print(f"  {k}: dict with {len(v)} keys")
        for sk in list(v.keys())[:50]:
            sv = v[sk]
            if hasattr(sv, "shape"):
                print(f"    {sk}: tensor shape={tuple(sv.shape)} dtype={sv.dtype}")
            else:
                print(f"    {sk}: {type(sv).__name__}")
    elif hasattr(v, "shape"):
        print(f"  {k}: tensor shape={tuple(v.shape)}")
    else:
        print(f"  {k}: {type(v).__name__} = {v if isinstance(v, (int, float, str)) else '...'}")
