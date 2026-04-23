from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator(r'C:\Users\nars7\Downloads\aniML_world\logs\rsl_rl\animl_deer_walk_p1\2026-04-14_17-32-16')
ea.Reload()
ep = ea.Scalars("Train/mean_episode_length")
ptl = ea.Scalars("Episode_Termination/pelvis_too_low")
sp = ea.Scalars("Episode_Reward/stationary_penalty")
ve = ea.Scalars("Episode_Reward/velocity_error")
vel = ea.Scalars("Episode_Reward/track_lin_vel_xy_exp")

print(f"FINAL: iter {ep[-1].step}  ep_len={ep[-1].value:.1f}  ptl={ptl[-1].value:.3f}")
print(f"MAX ep_len: {max(e.value for e in ep):.1f} at iter {max(ep, key=lambda e: e.value).step}")
print(f"\nLast 10:")
for i in range(-10, 0):
    print(f"  iter {ep[i].step:4d}  ep_len={ep[i].value:6.1f}  vel_xy={vel[i].value:.4f}  stationary={sp[i].value:+.4f}  vel_err={ve[i].value:+.4f}")
