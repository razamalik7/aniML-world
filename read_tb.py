import os, sys
logdir = r"C:\Users\nars7\Downloads\aniML_world\logs\rsl_rl\animl_deer_walk_p1\2026-04-13_22-13-01"
fname = os.path.join(logdir, "events.out.tfevents.1776143585.Polar.8352.0")
print(f"File size: {os.path.getsize(fname)} bytes", flush=True)
try:
    from tensorboard.backend.event_processing import event_accumulator
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()
    tags = ea.Tags()["scalars"]
    print("Tags:", tags[:10], flush=True)
    for tag in ["Train/mean_reward","Train/mean_episode_length","Train/mean_action_noise_std",
                "Episode_Termination/pelvis_too_low","Episode_Termination/time_out",
                "Episode_Reward/leg_extension","Episode_Reward/track_lin_vel_xy_exp"]:
        if tag in tags:
            evts = ea.Scalars(tag)
            sampled = [evts[i] for i in range(0, len(evts), max(1,len(evts)//15))]
            print(f"\n{tag}:")
            for e in sampled:
                print(f"  iter {e.step:5d}: {e.value:.4f}")
except Exception as ex:
    print(f"ERROR: {ex}", flush=True)
    import traceback; traceback.print_exc()
