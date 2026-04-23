import torch
ckpt = torch.load(r"C:\Users\nars7\Downloads\aniML_world\logs\rsl_rl\animl_deer_stand_v2\2026-04-13_22-09-04\model_199.pt", map_location="cpu", weights_only=False)
actor = ckpt["actor_state_dict"]
std_keys = [k for k in actor if "std" in k.lower()]
for k in std_keys:
    print(f"{k}: {actor[k]}")
print(f"iter: {ckpt.get('iter','?')}")
