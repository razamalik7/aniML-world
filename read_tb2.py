import os
from tensorboard.backend.event_processing import event_accumulator
logdir = r"C:\Users\nars7\Downloads\aniML_world\logs\rsl_rl\animl_deer_walk_p1\2026-04-13_22-13-01"
ea = event_accumulator.EventAccumulator(logdir)
ea.Reload()
tags = ea.Tags()["scalars"]
print("ALL TAGS:", tags)
for tag in ["Train/mean_action_noise_std","Episode_Reward/flat_orientation_l2",
            "Episode_Reward/base_height_l2","Episode_Reward/feet_air_time",
            "Curriculum/ramp_vel_xy","Curriculum/ramp_leg_ext"]:
    if tag in tags:
        evts = ea.Scalars(tag)
        sampled = [evts[i] for i in range(0, len(evts), max(1,len(evts)//15))]
        print(f"\n{tag}:")
        for e in sampled:
            print(f"  iter {e.step:5d}: {e.value:.5f}")
