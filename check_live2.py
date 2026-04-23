import torch
ckpt = torch.load(r'C:\Users\nars7\Downloads\aniML_world\logs\rsl_rl\animl_deer_walk_p1\2026-04-14_11-14-59\model_0.pt', map_location='cpu', weights_only=False)
print('LIVE model_0 actor keys:')
for k, v in sorted(ckpt['actor_state_dict'].items()):
    print(f'  {k}: {list(v.shape)}')
