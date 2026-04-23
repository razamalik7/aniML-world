import torch
ckpt = torch.load(r"C:\Users\nars7\Downloads\aniML_world\logs\rsl_rl\animl_deer_walk_p1\2026-04-14_11-14-59\model_0.pt", map_location="cpu", weights_only=False)
print("model_0.pt actor keys:")
for k in sorted(ckpt['actor_state_dict'].keys()):
    v = ckpt['actor_state_dict'][k]
    if 'distribution' in k or 'std' in k:
        print(f"  {k}: shape={v.shape}, val={v[:3] if v.numel()>1 else v.item():.5f}")
