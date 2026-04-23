import os
from tensorboard.backend.event_processing import event_accumulator

# Compare curriculum timing between the run that WORKED (April 13, 2026-04-13_15-58-29)
# and the current warm-start (2026-04-14_10-30-26)
for label, logdir in [
    ("WORKED (Apr13 iter500-1300)", r"C:\Users\nars7\Downloads\aniML_world\logs\rsl_rl\animl_deer_walk_p1\2026-04-13_15-58-29"),
    ("WARMSTART (Apr14)", r"C:\Users\nars7\Downloads\aniML_world\logs\rsl_rl\animl_deer_walk_p1\2026-04-14_10-30-26"),
]:
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()
    tags = ea.Tags()["scalars"]
    print(f"\n====== {label} ======")
    for tag in ["Curriculum/ramp_vel_xy","Curriculum/ramp_leg_ext","Train/mean_episode_length","Episode_Termination/pelvis_too_low","Policy/mean_std"]:
        if tag in tags:
            evts = ea.Scalars(tag)
            # show first 15 points
            sampled = evts[:15]
            print(f"  {tag}:")
            for e in sampled:
                print(f"    iter {e.step:5d}: {e.value:.4f}")
