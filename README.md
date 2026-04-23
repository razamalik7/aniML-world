# aniML World

Intelligent ecosystem simulation: train small PPO policies in **Isaac Lab**, deploy in **Unity 6** with Sentis. See `ARCHITECTURE.md` for tensor layouts, version pins, and the full directory map.

## Your machine (first-time)

1. Install **Isaac Sim** and **Isaac Lab** using NVIDIA’s matrix for your versions.
2. From this repo root, set `PYTHONPATH` to the repo root so `training.tasks` registers Gym environments, then run Lab’s trainer (example PowerShell):

```powershell
$env:PYTHONPATH = "C:\path\to\aniML_world"
$env:ISAACLAB_ROOT = "C:\isaac-lab\IsaacLab"
cd $env:ISAACLAB_ROOT
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task AniML-Deer-Locomotion-Rough-v0 --headless
```

Or use `tools/isaac_windows/Run-AniMLDeerTrain.ps1` after editing `config.ps1` if your paths differ.

3. **Isaac Lab assets** (`isaaclab_assets`, including ANYmal USD) must be available for the stand-in task **`AniML-Deer-Locomotion-Rough-v0`** (ANYmal-C under the hood).

4. **Your deer USD:** put `deer.usd` in `assets/deer/` or set **`ANIML_DEER_USD`** to an absolute path, then run **`tools/isaac_windows/Run-AniMLDeerUsdTrain.ps1`** (tasks **`AniML-Deer-USD-Rough-v0`** / **`AniML-Deer-USD-Flat-v0`**). See `assets/deer/PLACE_DEER_USD_HERE.txt` and `training/envs/deer_usd_rough_env_cfg.py`.

## Repo-only tooling

```powershell
python shared/schemas.py --export
```

Regenerates `shared/schemas.json` from `shared/schemas.py`.

## Windows helpers

See `tools/isaac_windows/` and Cursor **Tasks** (Fix Kit Python, verify empty scene, Cartpole smoke, aniML deer train).
