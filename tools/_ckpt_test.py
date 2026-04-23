import sys
sys.path.insert(0, r'C:\Users\nars7\Downloads\aniML_world')
sys.path.insert(0, r'C:\isaac-lab\IsaacLab\scripts\reinforcement_learning\rsl_rl')
from isaaclab_tasks.utils import get_checkpoint_path
p = get_checkpoint_path(
    r'C:\Users\nars7\Downloads\aniML_world\logs\rsl_rl\animl_deer_walk_p1',
    '2026-04-14_12-55-18',
    'model_3700.pt'
)
print("Resolved path:", p)
