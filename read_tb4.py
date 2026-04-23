import os
from tensorboard.backend.event_processing import event_accumulator

# Also check the early April 10 runs to see what configs produced different behavior
# Check the longest-running one: 2026-04-10_18-55-21
logdir = r"C:\Users\nars7\Downloads\aniML_world\logs\rsl_rl\animl_deer_walk_p1\2026-04-10_18-55-21"
if not os.path.exists(logdir):
    print("Not found:", logdir)
else:
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()
    tags = ea.Tags()["scalars"]
    for tag in ["Train/mean_episode_length","Episode_Termination/pelvis_too_low","Policy/mean_std","Episode_Reward/leg_extension"]:
        if tag in tags:
            evts = ea.Scalars(tag)
            sampled = [evts[i] for i in range(0, len(evts), max(1,len(evts)//12))]
            print(f"\n{tag}:")
            for e in sampled:
                print(f"  iter {e.step:5d}: {e.value:.4f}")

print("\n\n--- checking params dir of fresh run ---")
params_dir = r"C:\Users\nars7\Downloads\aniML_world\logs\rsl_rl\animl_deer_walk_p1\2026-04-13_22-13-01\params"
for f in os.listdir(params_dir):
    print(f)
