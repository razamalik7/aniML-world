"""Rough VRAM planning helper (RTX 5070 12GB — see ARCHITECTURE.md)."""

from __future__ import annotations


def estimate_training_gb(*, num_envs: int, mb_per_env: float = 4.5) -> float:
    """Heuristic only; profile real runs in Isaac Lab."""
    return num_envs * mb_per_env / 1024.0


def main() -> None:
    for n in (512, 1024, 2048, 4096):
        print(f"num_envs={n} ~ {estimate_training_gb(num_envs=n):.1f} GB (heuristic)")


if __name__ == "__main__":
    main()
