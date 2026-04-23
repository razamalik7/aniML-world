import os
from tensorboard.backend.event_processing import event_accumulator

logbase = r"C:\Users\nars7\Downloads\aniML_world\logs\rsl_rl\animl_deer_walk_p1"
runs = sorted([d for d in os.listdir(logbase) if os.path.isdir(os.path.join(logbase, d)) and d.startswith("2026")])

print(f"{'Run':<30} {'MaxEpLen':>10} {'AtIter':>8} {'FinalEpLen':>12} {'FinalPTL':>12}")
print("-"*80)
for run in runs:
    logdir = os.path.join(logbase, run)
    try:
        ea = event_accumulator.EventAccumulator(logdir)
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        if "Train/mean_episode_length" not in tags:
            continue
        el = ea.Scalars("Train/mean_episode_length")
        ptl = ea.Scalars("Episode_Termination/pelvis_too_low") if "Episode_Termination/pelvis_too_low" in tags else []
        max_el = max(e.value for e in el)
        max_iter = next(e.step for e in el if e.value == max_el)
        final_el = el[-1].value
        final_ptl = ptl[-1].value if ptl else -1
        print(f"{run:<30} {max_el:>10.1f} {max_iter:>8d} {final_el:>12.1f} {final_ptl:>12.3f}")
    except Exception as ex:
        print(f"{run:<30} ERROR: {ex}")
