import os
from tensorboard.backend.event_processing import event_accumulator

logdir = r"C:\Users\nars7\Downloads\aniML_world\logs\rsl_rl\animl_deer_walk_p1\2026-04-13_22-13-01"
ea = event_accumulator.EventAccumulator(logdir)
ea.Reload()
tags = ea.Tags()["scalars"]

for tag in ["Policy/mean_std", "Loss/learning_rate", "Loss/entropy"]:
    if tag in tags:
        evts = ea.Scalars(tag)
        sampled = [evts[i] for i in range(0, len(evts), max(1,len(evts)//15))]
        print(f"\n{tag}:")
        for e in sampled:
            print(f"  iter {e.step:5d}: {e.value:.5f}")
