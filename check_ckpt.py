import torch
ckpt = torch.load(r'C:\Users\nars7\Downloads\aniML_world\logs\rsl_rl\animl_deer_walk_p1\2026-04-13_15-58-29\model_800.pt', map_location='cpu')
print('iter:', ckpt['iter'])
print('infos:', ckpt.get('infos', {}))
print('std:', ckpt['actor_state_dict'].get('distribution.std_param', 'N/A'))
