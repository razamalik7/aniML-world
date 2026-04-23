import torch
ckpt = torch.load(r'C:\Users\nars7\Downloads\aniML_world\logs\rsl_rl\animl_deer_walk_p1\2026-04-14_11-14-59\model_0.pt', map_location='cpu', weights_only=False)
std = ckpt['actor_state_dict']['distribution.std_param']
print(f'model_0.pt std_param: {std.tolist()}')
print(f'mean std: {std.mean().item():.4f}')

# Also check the loaded checkpoint
src = torch.load(r'C:\Users\nars7\Downloads\aniML_world\logs\rsl_rl\animl_deer_walk_p1\stand_v2_walk_init\model_199_raw.pt', map_location='cpu', weights_only=False)
src_std = src['actor_state_dict']['distribution.std_param']
print(f'source checkpoint std_param: {src_std.tolist()}')
print(f'mean src std: {src_std.mean().item():.4f}')
