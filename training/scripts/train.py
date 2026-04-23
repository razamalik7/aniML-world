"""
Register aniML Gym tasks, then run Isaac Lab's RSL-RL trainer.

Usage (PowerShell, from Isaac Lab root):

  $env:PYTHONPATH = 'C:\\path\\to\\aniML_world'
  $env:ISAACLAB_ROOT = 'C:\\isaac-lab\\IsaacLab'
  .\\isaaclab.bat -p C:\\path\\to\\aniML_world\\training\\scripts\\train.py `
      --task AniML-Deer-Locomotion-Rough-v0 --headless

Arguments after this script are forwarded to ``scripts/reinforcement_learning/rsl_rl/train.py``.
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


def _lab_train_script() -> Path:
    root = Path(os.environ.get("ISAACLAB_ROOT", r"C:\isaac-lab\IsaacLab")).resolve()
    script = root / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py"
    if not script.is_file():
        raise FileNotFoundError(
            f"Isaac Lab train.py not found at {script}. Set ISAACLAB_ROOT to your Isaac Lab clone."
        )
    return script


def main() -> None:
    _ensure_path()
    import training.tasks  # noqa: F401 — side effect: gym.register

    target = _lab_train_script()
    # runpy.run_path() does not put the script directory on sys.path for a .py file
    # (Python 3.12+), so Isaac's `import cli_args` would fail without this.
    rsl_rl_dir = str(target.resolve().parent)
    if rsl_rl_dir not in sys.path:
        sys.path.insert(0, rsl_rl_dir)

    forwarded = [str(target), *sys.argv[1:]]
    sys.argv = forwarded
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
