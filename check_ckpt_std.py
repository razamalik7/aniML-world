import torch

for ckpt in ["model_50.pt", "model_100.pt", "model_150.pt", "model_199.pt"]:
    path = rf"C:\Users\nars7\Downloads\aniML_world\logs\rsl_rl\animl_deer_stand_v2\2026-04-13_22-09-04\{ckpt}"
    d = torch.load(path, map_location="cpu", weights_only=False)
    keys = list(d.keys())
    # get std param if present
    actor = d.get("actor_state_dict") or d.get("model_state_dict") or {}
    std_keys = [k for k in actor if "std" in k.lower() or "log_std" in k.lower()]
    std_vals = {k: actor[k].item() if actor[k].numel()==1 else actor[k].mean().item() for k in std_keys}
    print(f"\n{ckpt}: top-level keys={keys}, std_params={std_vals}")
