"""Load a trained checkpoint and visualize it in Isaac Sim.

Usage (PowerShell, from Isaac Lab root):

  $env:PYTHONPATH = 'C:\\path\\to\\aniML_world'
  .\\isaaclab.bat -p C:\\path\\to\\aniML_world\\training\\scripts\\play.py `
      --task AniML-Deer-USD-Stand-v0 --num_envs 32
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_path() -> None:
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _lab_play_script() -> Path:
    root = Path(os.environ.get("ISAACLAB_ROOT", r"C:\isaac-lab\IsaacLab")).resolve()
    script = root / "scripts" / "reinforcement_learning" / "rsl_rl" / "play.py"
    if not script.is_file():
        raise FileNotFoundError(
            f"Isaac Lab play.py not found at {script}. Set ISAACLAB_ROOT."
        )
    return script


def main() -> None:
    _ensure_path()
    import training.tasks  # noqa: F401

    target = _lab_play_script()
    rsl_rl_dir = str(target.resolve().parent)
    if rsl_rl_dir not in sys.path:
        sys.path.insert(0, rsl_rl_dir)

    forwarded = [str(target), *sys.argv[1:]]
    sys.argv = forwarded
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
