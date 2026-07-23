"""Generate the corrupted-reference suite for the AMP noise-tolerance experiment.

    python tools/cv/make_noise_suite.py --write

Writes `assets/motions/noise_suite/` plus a `MANIFEST.md` describing every
variant and the prediction it tests. CPU-only; the training runs themselves need
the GPU.

WHY THIS EXPERIMENT
-------------------
The accuracy bar for every stage of the video->motion pipeline is set by one
number nobody has published: **how much reference-motion noise can AMP absorb
before the policy degrades?** A literature sweep found no study that deliberately
corrupts a reference and measures AMP policy quality against corruption level.
The closest (I-CTRL) is tracking-based and applies noise at evaluation time.

So the curve has to be measured here. It is cheap -- six short runs -- and it
either tightens or dramatically relaxes the requirements on the 2D pose net, the
URDF fit, and the smoothing stage.

THE PREDICTION BEING TESTED
---------------------------
`amp-rsl-rl` linearly resamples the clip and then takes a **raw finite
difference** -- no smoothing anywhere -- to build the joint-velocity half of the
36-dim discriminator observation. Jitter is therefore amplified by
`sqrt(2) * fps` ~= 42x at 30 fps. On this project's own walk clip (true RMS joint
speed 1.63 rad/s):

| jitter sigma | induced joint-vel std | % of true signal |
|---|---|---|
| 0.5 deg | 0.37 rad/s | 23 % |
| 1 deg | 0.74 rad/s | **45 %** |
| 2 deg | 1.48 rad/s | 91 % |
| 3 deg | 2.22 rad/s | 136 % |
| 5 deg | 3.70 rad/s | 227 % |

A physically actuated deer cannot produce white-noise joint velocity, so a noisy
velocity channel should become a perfectly separating feature: discriminator
saturates, style reward collapses to ~0. **Predicted knee somewhere between 1 and
2 degrees.** If instead style reward survives 5 deg, the whole upstream accuracy
budget relaxes and the pipeline gets much cheaper.

Each variant also ships a `--smoothed` twin (Butterworth zero-phase, 6 Hz) so the
experiment simultaneously measures how much of the damage filtering recovers --
which is the actual design question, since the pipeline will filter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
SRC = REPO / "assets" / "motions" / "deer_walk.npy"
OUT = REPO / "assets" / "motions" / "noise_suite"

JITTER_DEG = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
SPEED_SCALES = [0.5, 2.0]
LIMB_SCALE = 1.10


def load(path: Path) -> dict:
    return np.load(path, allow_pickle=True).item()


def butter_lowpass_zero_phase(x: np.ndarray, fps: float,
                              cutoff_hz: float = 6.0) -> np.ndarray:
    """Zero-phase low-pass. Falls back to a Hann FIR if SciPy is absent.

    Zero-phase matters: a causal filter shifts the signal in time, which changes
    the gait's phase relationships. Those are exactly what the discriminator
    looks at.
    """
    try:
        from scipy.signal import butter, filtfilt  # type: ignore
        b, a = butter(4, cutoff_hz / (0.5 * fps), btype="low")
        pad = min(len(x) - 1, 3 * max(len(a), len(b)))
        return filtfilt(b, a, x, axis=0, padlen=pad)
    except Exception:
        n = max(3, int(round(fps / cutoff_hz)) | 1)
        w = np.hanning(n + 2)[1:-1]
        w /= w.sum()
        pad = n // 2
        xp = np.pad(x, ((pad, pad), (0, 0)), mode="wrap")
        return np.stack([np.convolve(xp[:, i], w, mode="valid")
                         for i in range(x.shape[1])], axis=1)


def cyclic_velocity(x: np.ndarray, fps: float) -> np.ndarray:
    T = x.shape[0]
    last = T - 2 if np.allclose(x[0], x[-1], atol=1e-4) else T - 1
    period = last + 1
    dt = 1.0 / fps
    out = np.zeros_like(x)
    for f in range(T):
        out[f] = (x[(f + 1) % period] - x[(f - 1) % period]) / (2 * dt)
    return out


def loader_side_velocity(x: np.ndarray, fps: float) -> np.ndarray:
    """Velocity as `amp-rsl-rl` will actually compute it.

    The loader IGNORES the `joint_velocities` stored in the file. It linearly
    resamples `joint_positions` to the sim timestep and applies a bare FORWARD
    difference (`_compute_raw_derivative`), with no smoothing. Linear
    interpolation makes the derivative piecewise-constant at the SOURCE frame
    rate, so the effective operation is a forward difference at `fps`.

    This matters for the experiment: a forward difference amplifies iid noise by
    `sqrt(2)/dt`, twice the `sqrt(2)/(2 dt)` of the central difference used when
    writing the clip. Reporting the stored velocity would understate what the
    discriminator sees by 2x.
    """
    dt = 1.0 / fps
    return (np.roll(x, -1, axis=0) - x) / dt


def write_variant(out_dir: Path, name: str, base: dict, jp: np.ndarray,
                  rp: np.ndarray, write: bool) -> dict:
    fps = float(np.asarray(base["fps"]).reshape(-1)[0])
    jv = cyclic_velocity(jp, fps)
    payload = {
        "joints_list": list(base["joints_list"]),
        "joint_positions": jp.astype(np.float32),
        "joint_velocities": jv.astype(np.float32),
        "root_position": rp.astype(np.float32),
        "root_quaternion": np.asarray(base["root_quaternion"], np.float32),
        "fps": fps,
    }
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / f"{name}.npy", payload, allow_pickle=True)
    jv_loader = loader_side_velocity(jp, fps)
    return {
        "name": name,
        "joint_vel_rms": float(np.sqrt((jv ** 2).mean())),
        "loader_vel_rms": float(np.sqrt((jv_loader ** 2).mean())),
        "joint_pos_rms": float(np.sqrt((jp ** 2).mean())),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    base = load(Path(args.src))
    fps = float(np.asarray(base["fps"]).reshape(-1)[0])
    jp0 = np.asarray(base["joint_positions"], np.float64)
    rp0 = np.asarray(base["root_position"], np.float64)
    out_dir = Path(args.out)

    true_vel_rms = float(np.sqrt((loader_side_velocity(jp0, fps) ** 2).mean()))
    rows: list[dict] = []
    manifest: list[str] = [
        "# AMP noise-tolerance suite",
        "",
        f"Source: `{Path(args.src).relative_to(REPO)}` "
        f"({jp0.shape[0]} frames @ {fps:g} fps, {jp0.shape[1]} joints)",
        f"Seed: {args.seed}. True RMS joint speed: **{true_vel_rms:.3f} rad/s**",
        "",
        "Train one short AMP run per variant, then plot style reward and the",
        "`amp_status.py` metrics against the corruption level.",
        "",
        "| variant | corruption | induced joint-vel RMS | vs true |",
        "|---|---|---|---|",
    ]

    # --- jitter sweep, raw and smoothed ---
    for deg in JITTER_DEG:
        sigma = np.deg2rad(deg)
        noise = rng.normal(0.0, sigma, jp0.shape) if sigma > 0 else 0.0
        jp = jp0 + noise
        r = write_variant(out_dir, f"jitter_{deg:g}deg", base, jp, rp0,
                          args.write)
        rows.append(r)
        manifest.append(
            f"| `jitter_{deg:g}deg` | iid Gaussian, sigma={deg:g} deg | "
            f"{r['loader_vel_rms']:.3f} rad/s | "
            f"{r['loader_vel_rms']/true_vel_rms*100:.0f}% |")
        if deg > 0:
            jps = butter_lowpass_zero_phase(jp, fps, 6.0)
            rs = write_variant(out_dir, f"jitter_{deg:g}deg_smoothed", base,
                               jps, rp0, args.write)
            rows.append(rs)
            manifest.append(
                f"| `jitter_{deg:g}deg_smoothed` | same + Butterworth 6 Hz "
                f"zero-phase | {rs['loader_vel_rms']:.3f} rad/s | "
                f"{rs['loader_vel_rms']/true_vel_rms*100:.0f}% |")

    # --- systematic errors ---
    for s in SPEED_SCALES:
        rp = rp0.copy()
        rp[:, :2] = rp0[:, :2] * s
        r = write_variant(out_dir, f"rootspeed_x{s:g}", base, jp0, rp,
                          args.write)
        rows.append(r)
        manifest.append(
            f"| `rootspeed_x{s:g}` | root speed x{s:g}, legs unchanged "
            f"(contact-schedule violation) | {r['loader_vel_rms']:.3f} rad/s "
            f"| 100% |")

    jp_scaled = jp0 * LIMB_SCALE
    r = write_variant(out_dir, "jointscale_x1.1", base, jp_scaled, rp0,
                      args.write)
    rows.append(r)
    manifest.append(
        f"| `jointscale_x1.1` | all joint angles x1.10 (systematic bias, no "
        f"noise) | {r['loader_vel_rms']:.3f} rad/s | "
        f"{r['loader_vel_rms']/true_vel_rms*100:.0f}% |")

    manifest += [
        "",
        "## What each variant tests",
        "",
        "- **jitter_\\*** -- the core dose-response curve. Prediction: style",
        "  reward collapses somewhere between 1 and 2 deg, because the induced",
        "  joint-velocity noise crosses ~50% of the true signal there and the",
        "  policy cannot physically reproduce white-noise velocity.",
        "- **jitter_\\*_smoothed** -- how much a zero-phase 6 Hz Butterworth",
        "  recovers. This is the pair that actually sets the pipeline's budget:",
        "  if smoothing rescues 3 deg, the 2D pose net can be much sloppier.",
        "- **rootspeed_x\\*** -- contact-schedule violation with clean joints.",
        "  Tests whether foot-slide/root-speed inconsistency poisons AMP the way",
        "  it demonstrably poisons tracking-based imitation. This one is a",
        "  genuine unknown; all the supporting evidence in the literature is",
        "  from tracking methods.",
        "- **jointscale_x1.1** -- systematic bias with zero noise. Physics should",
        "  absorb this (it stays on the feasible manifold); expect a stable but",
        "  slightly odd gait rather than collapse. A control for 'is it noise or",
        "  is it just being wrong'.",
        "",
        "## Reading the result",
        "",
        "Watch `style_r/step`. The documented failure fingerprint is style",
        "reward collapsing to ~0 while the discriminator saturates near 100%",
        "accuracy (`CLAUDE.md`, 'AMP failure-mode fingerprints'). Compare each",
        "variant against `jitter_0deg`, which is the uncorrupted control and",
        "should reproduce whatever the current baseline does.",
        "",
        "Keep every other knob fixed across runs -- same seed, same iterations,",
        "same env cfg. The point is a curve, not a best result.",
    ]

    print(f"source RMS joint speed: {true_vel_rms:.3f} rad/s")
    for r in rows:
        print(f"  {r['name']:<32} loader-side vel RMS {r['loader_vel_rms']:6.3f} "
              f"({r['loader_vel_rms']/true_vel_rms*100:5.0f}% of true)")
    if args.write:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "MANIFEST.md").write_text("\n".join(manifest),
                                             encoding="utf-8")
        (out_dir / "suite.json").write_text(
            json.dumps({"source": str(args.src), "seed": args.seed,
                        "true_vel_rms": true_vel_rms, "variants": rows},
                       indent=1), encoding="utf-8")
        print(f"\nwrote {len(rows)} variants + MANIFEST.md to {out_dir}")
    else:
        print("\n(dry run -- pass --write)")


if __name__ == "__main__":
    main()
