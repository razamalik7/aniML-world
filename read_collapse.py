import os
from tensorboard.backend.event_processing import event_accumulator

# Find most recent walk run
logbase = r"C:\Users\nars7\Downloads\aniML_world\logs\rsl_rl\animl_deer_walk_p1"
runs = sorted([d for d in os.listdir(logbase) if d.startswith("2026-04-14")])
logdir = os.path.join(logbase, runs[-1])
print(f"Reading: {logdir}")

ea = event_accumulator.EventAccumulator(logdir)
ea.Reload()
tags = ea.Tags()["scalars"]

# Get dense view of first 120 iters
for tag in ["Train/mean_episode_length","Episode_Termination/pelvis_too_low","Policy/mean_std","Loss/entropy"]:
    if tag in tags:
        evts = ea.Scalars(tag)
        pts = [e for e in evts if e.step <= 120]
        print(f"\n{tag}:")
        for e in pts[::5]:  # every 5 iters
            print(f"  iter {e.step:4d}: {e.value:.4f}")
