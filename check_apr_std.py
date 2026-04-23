import torch
ckpt = torch.load(r"C:\Users\nars7\Downloads\aniML_world\logs\rsl_rl\animl_deer_walk_p1\warmstart_from_stand_v2\model_499.pt", map_location="cpu", weights_only=False)
actor = ckpt["actor_state_dict"]
std_keys = [k for k in actor if "std" in k.lower()]
for k in std_keys:
    print(f"{k}: {actor[k]}")
print(f"iter: {ckpt.get('iter','?')}")
