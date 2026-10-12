"""Compare AMP runs on the metrics that actually distinguish them.

Built because the single-number view (tools/amp_status.py, last-20 average) makes
runs at different ages look comparable when they are not: Exp 1's amp_loss of 0.243
at iteration 53 and r16p1's 0.299 at iteration 1559 describe completely different
situations. Discriminator saturation is a TRAJECTORY, not a level.

Reads amp_loss / style_r / ep_len at matched iterations across runs.

  python tools/compare_runs.py <run_dir_or_substr> [<run_dir_or_substr> ...]
  python tools/compare_runs.py --list
"""

from __future__ import annotations

import sys
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "logs" / "rsl_rl" / "animl_deer_polyart_walk"

TAGS = {
    "amp_loss": ["Loss/amp_loss"],
    "policy_pred": ["Loss/policy_pred"],
    "expert_pred": ["Loss/expert_pred"],
    "style_r": ["Train/mean_style_reward", "Train/style_reward"],
    "ep_len": ["Train/mean_episode_length", "Train/mean_episode_length/time"],
}


def load(run: Path) -> dict:
    ea = event_accumulator.EventAccumulator(str(run), size_guidance={event_accumulator.SCALARS: 20000})
    ea.Reload()
    have = ea.Tags()["scalars"]
    out = {}
    for key, cands in TAGS.items():
        for c in cands:
            if c in have:
                out[key] = {e.step: e.value for e in ea.Scalars(c)}
                break
    out["_tags"] = have
    return out


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    runs = sorted([p for p in ROOT.iterdir() if p.is_dir()])
    if "--list" in sys.argv or not args:
        for p in runs:
            print(" ", p.name)
        if not args:
            return
    sel = []
    for a in args:
        m = [p for p in runs if a in p.name]
        if not m:
            print(f"no run matching {a!r}"); continue
        sel.append(m[-1])
    data = {p.name: load(p) for p in sel}
    if not data:
        return
    if "amp_loss" not in next(iter(data.values())):
        print("available scalar tags:", next(iter(data.values()))["_tags"][:25]); return

    marks = [25, 50, 100, 200, 400, 700, 1000, 1500, 2000, 2500]
    for metric in ("amp_loss", "policy_pred", "expert_pred", "style_r", "ep_len"):
        print(f"\n=== {metric} ===")
        print("%-26s" % "run" + "".join("%9s" % m for m in marks))
        print("-" * (26 + 9 * len(marks)))
        for name, d in data.items():
            series = d.get(metric, {})
            row = ""
            for m in marks:
                near = [s for s in series if abs(s - m) <= max(5, m * 0.03)]
                row += "%9.3f" % series[min(near, key=lambda s: abs(s - m))] if near else "%9s" % "-"
            print("%-26s%s" % (name[:26], row))
    print("\nln(2) = 0.693 is the amp_loss equilibrium: discriminator CANNOT separate")
    print("policy from expert. Falling well below it = saturation, style reward decays.")


if __name__ == "__main__":
    main()
