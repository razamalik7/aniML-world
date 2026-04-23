import sys, inspect, importlib
sys.path.insert(0, r'C:/isaac-lab/IsaacLab/source')
try:
    m = importlib.import_module('isaaclab_rl.rsl_rl.runners.on_policy_runner')
    src = inspect.getsource(m)
    idx = src.find('load_run')
    print(src[max(0,idx-300):idx+1000])
except Exception as e:
    print(f"Error: {e}")
    # fallback: find the file
    import glob
    files = glob.glob(r'C:/isaac-lab/IsaacLab/**/*runner*', recursive=True)
    print(files[:10])
