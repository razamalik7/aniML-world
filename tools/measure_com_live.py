"""
Spawn deer_standing_v2.usd in a live Isaac Lab sim and measure actual body heights.
Run with: isaaclab.bat -p tools/measure_com_live.py
"""
import torch
from isaaclab.app import AppLauncher

_out = open(r'C:\Users\nars7\Downloads\aniML_world\logs\com_results.txt', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a)
    import builtins
    builtins.print(s)
    _out.write(s + '\n')
    _out.flush()

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
import sys
sys.path.insert(0, r'C:\Users\nars7\Downloads\aniML_world')
from training.envs.deer_usd_articulation import deer_articulation_cfg
from training.envs.deer_usd_rough_env_cfg import _require_usd_path

sim_cfg = sim_utils.SimulationCfg(dt=1/60, device="cpu")
sim = SimulationContext(sim_cfg)
sim.set_camera_view([3.0, 0.0, 3.0], [0.0, 0.0, 1.0])

sim_utils.GroundPlaneCfg().func("/World/GroundPlane", sim_utils.GroundPlaneCfg())

usd_path = _require_usd_path()
P(f"\n[measure_com] USD: {usd_path}")

deer_cfg = deer_articulation_cfg(usd_path)
deer_cfg.prim_path = "/World/Deer"
deer = Articulation(deer_cfg)

sim.reset()
deer.reset()

P(f"[measure_com] Stepping 120 frames to let deer settle...")
for _ in range(120):
    deer.write_data_to_sim()
    sim.step()
    deer.update(sim.get_physics_dt())

body_names = list(deer.data.body_names)
body_pos   = deer.data.body_pos_w[0]  # (N, 3)

P(f"\nBody positions in Isaac world (Z=up):")
P(f"  {'Name':<25} {'X':>8} {'Y':>8} {'Z_height':>10}")
P(f"  {'-'*55}")

foot_names = ['HR_foot', 'HL_foot', 'FR_foot', 'FL_foot']
foot_z = []
for i, name in enumerate(body_names):
    pos = body_pos[i]
    marker = ' << FOOT' if name in foot_names else ''
    if name == 'pelvis': marker = ' << PELVIS'
    if name == 'chest':  marker = ' << CHEST'
    if name == 'skull':  marker = ' << SKULL'
    P(f"  {name:<25} {pos[0].item():>8.4f} {pos[1].item():>8.4f} {pos[2].item():>10.4f}{marker}")
    if name in foot_names:
        foot_z.append(pos[2].item())

avg_foot = sum(foot_z) / len(foot_z) if foot_z else 0.
pelvis_i = body_names.index('pelvis') if 'pelvis' in body_names else -1
chest_i  = body_names.index('chest')  if 'chest'  in body_names else -1
skull_i  = body_names.index('skull')  if 'skull'  in body_names else -1

pelvis_h = body_pos[pelvis_i][2].item() if pelvis_i >= 0 else float('nan')
chest_h  = body_pos[chest_i][2].item()  if chest_i  >= 0 else float('nan')
skull_h  = body_pos[skull_i][2].item()  if skull_i  >= 0 else float('nan')

P(f"\nKey heights:")
P(f"  Avg foot Z  = {avg_foot:.4f} m  (should be ~0 = on ground)")
P(f"  Pelvis Z    = {pelvis_h:.4f} m")
P(f"  Chest Z     = {chest_h:.4f} m")
P(f"  Skull Z     = {skull_h:.4f} m")
P(f"  Pelvis above feet = {pelvis_h - avg_foot:.4f} m")
P(f"  Chest above feet  = {chest_h  - avg_foot:.4f} m")
P(f"  Skull above feet  = {skull_h  - avg_foot:.4f} m")

# Estimate CoM from body positions (unweighted - just geometric center of rigid bodies)
geom_com_z = body_pos[:, 2].mean().item()
P(f"\n  Geometric center Z (all bodies) = {geom_com_z:.4f} m")
P(f"  => above feet: {geom_com_z - avg_foot:.4f} m")

# Try to get masses
P(f"\nAttempting mass-weighted CoM...")
masses = None
for attr in ['body_mass', 'default_mass']:
    val = getattr(deer.data, attr, None)
    if val is not None and hasattr(val, 'numel') and val.numel() > 1:
        masses = val[0]
        P(f"  Got masses from deer.data.{attr}: {masses.tolist()}")
        break

if masses is not None and len(masses) == len(body_names):
    total_m = masses.sum().item()
    com = (masses.unsqueeze(1) * body_pos).sum(0) / total_m
    P(f"\n  Total mass: {total_m:.2f} kg")
    P(f"  Mass-weighted CoM: X={com[0]:.4f} Y={com[1]:.4f} Z={com[2]:.4f}")
    P(f"  CoM above feet:   {com[2].item() - avg_foot:.4f} m")
    P(f"  CoM above pelvis: {com[2].item() - pelvis_h:.4f} m")
    P(f"  Real deer CoM above ground: ~0.65m")
    
    P(f"\n  Per-body masses:")
    items = [(masses[i].item(), body_names[i], body_pos[i][2].item()) for i in range(len(body_names))]
    for m, n, h in sorted(items, key=lambda x: -x[0]):
        P(f"    {n:<25} {m:>8.3f} kg  Z={h:.3f}")
else:
    P(f"  Mass data unavailable. Body count={len(body_names)}, masses shape={masses.shape if masses is not None else 'None'}")

P(f"\nDONE.")
_out.close()
simulation_app.close()
