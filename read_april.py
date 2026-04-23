import os
from tensorboard.backend.event_processing import event_accumulator

# Check the April run that stayed stable at 0.27-0.35 pelvis_too_low for 400 iters
logdir = r"C:\Users\nars7\Downloads\aniML_world\logs\rsl_rl\animl_deer_walk_p1\2026-04-13_15-58-29"
ea = event_accumulator.EventAccumulator(logdir)
ea.Reload()
tags = ea.Tags()["scalars"]

print("=== APRIL RUN THAT WORKED (first 60 iters from its start at iter 499) ===")
for tag in ["Train/mean_episode_length","Episode_Termination/pelvis_too_low","Episode_Termination/base_contact","Policy/mean_std","Episode_Reward/leg_extension","Loss/entropy"]:
    if tag in tags:
        evts = ea.Scalars(tag)
        # first 60 data points
        pts = evts[:60:5]
        print(f"\n{tag}:")
        for e in pts:
            print(f"  iter {e.step:5d}: {e.value:.4f}")
