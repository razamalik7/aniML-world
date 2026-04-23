import torch, sys
sys.path.insert(0, r"C:\isaac-lab\IsaacLab\_isaac_sim\kit\python\Lib\site-packages")

# Check what keys the current rsl_rl GaussianDistribution model uses
from rsl_rl.modules.actor_critic import MLPModel
from rsl_rl.utils.distributions import GaussianDistribution
import torch.nn as nn

class FakeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.obs_normalizer = nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(57, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 15),
        )
        self.distribution = GaussianDistribution(num_actions=15, init_std=1.0, std_type="scalar")
    def forward(self, x):
        return self.mlp(self.obs_normalizer(x))

m = FakeMLP()
print("Model distribution keys:")
for k, v in m.distribution.state_dict().items():
    print(f"  {k}: shape={v.shape}, val={v}")

# Now try loading checkpoint
ckpt = torch.load(r"C:\Users\nars7\Downloads\aniML_world\logs\rsl_rl\animl_deer_walk_p1\stand_v2_walk_init\model_199_raw.pt", map_location="cpu", weights_only=False)
print("\nCheckpoint distribution keys:")
for k, v in ckpt["actor_state_dict"].items():
    if "distribution" in k or "std" in k:
        print(f"  {k}: shape={v.shape}, val={v}")
