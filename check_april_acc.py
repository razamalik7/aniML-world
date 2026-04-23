import os
from tensorboard.backend.event_processing import event_accumulator
logdir = r"C:\Users\nars7\Downloads\aniML_world\logs\rsl_rl\animl_deer_walk_p1\2026-04-13_15-58-29"
ea = event_accumulator.EventAccumulator(logdir)
ea.Reload()
tags = ea.Tags()["scalars"]
for tag in ["Episode_Reward/dof_acc_l2","Episode_Reward/leg_extension","Train/mean_episode_length","Episode_Termination/pelvis_too_low"]:
    if tag in tags:
        evts = ea.Scalars(tag)
        pts = evts[:12]
        print(f"\n{tag}:")
        for e in pts:
            print(f"  iter {e.step:5d}: {e.value:.4f}")
