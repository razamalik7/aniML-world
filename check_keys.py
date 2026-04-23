import torch
ckpt = torch.load(r"C:\Users\nars7\Downloads\aniML_world\logs\rsl_rl\animl_deer_walk_p1\stand_v2_walk_init\model_199_raw.pt", map_location="cpu", weights_only=False)
print("Checkpoint actor keys:")
for k in sorted(ckpt["actor_state_dict"].keys()):
    v = ckpt["actor_state_dict"][k]
    print(f"  {k}: shape={v.shape}")
