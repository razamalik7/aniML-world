# AMP noise-tolerance suite

Source: `assets\motions\deer_walk.npy` (31 frames @ 30 fps, 15 joints)
Seed: 0. True RMS joint speed: **1.607 rad/s**

Train one short AMP run per variant, then plot style reward and the
`amp_status.py` metrics against the corruption level.

| variant | corruption | induced joint-vel RMS | vs true |
|---|---|---|---|
| `jitter_0deg` | iid Gaussian, sigma=0 deg | 1.607 rad/s | 100% |
| `jitter_0.5deg` | iid Gaussian, sigma=0.5 deg | 1.650 rad/s | 103% |
| `jitter_0.5deg_smoothed` | same + Butterworth 6 Hz zero-phase | 1.588 rad/s | 99% |
| `jitter_1deg` | iid Gaussian, sigma=1 deg | 1.736 rad/s | 108% |
| `jitter_1deg_smoothed` | same + Butterworth 6 Hz zero-phase | 1.591 rad/s | 99% |
| `jitter_2deg` | iid Gaussian, sigma=2 deg | 2.234 rad/s | 139% |
| `jitter_2deg_smoothed` | same + Butterworth 6 Hz zero-phase | 1.704 rad/s | 106% |
| `jitter_3deg` | iid Gaussian, sigma=3 deg | 2.804 rad/s | 174% |
| `jitter_3deg_smoothed` | same + Butterworth 6 Hz zero-phase | 1.746 rad/s | 109% |
| `jitter_5deg` | iid Gaussian, sigma=5 deg | 4.198 rad/s | 261% |
| `jitter_5deg_smoothed` | same + Butterworth 6 Hz zero-phase | 1.984 rad/s | 123% |
| `rootspeed_x0.5` | root speed x0.5, legs unchanged (contact-schedule violation) | 1.607 rad/s | 100% |
| `rootspeed_x2` | root speed x2, legs unchanged (contact-schedule violation) | 1.607 rad/s | 100% |
| `jointscale_x1.1` | all joint angles x1.10 (systematic bias, no noise) | 1.768 rad/s | 110% |

## What each variant tests

- **jitter_\*** -- the core dose-response curve. Prediction: style
  reward collapses somewhere between 1 and 2 deg, because the induced
  joint-velocity noise crosses ~50% of the true signal there and the
  policy cannot physically reproduce white-noise velocity.
- **jitter_\*_smoothed** -- how much a zero-phase 6 Hz Butterworth
  recovers. This is the pair that actually sets the pipeline's budget:
  if smoothing rescues 3 deg, the 2D pose net can be much sloppier.
- **rootspeed_x\*** -- contact-schedule violation with clean joints.
  Tests whether foot-slide/root-speed inconsistency poisons AMP the way
  it demonstrably poisons tracking-based imitation. This one is a
  genuine unknown; all the supporting evidence in the literature is
  from tracking methods.
- **jointscale_x1.1** -- systematic bias with zero noise. Physics should
  absorb this (it stays on the feasible manifold); expect a stable but
  slightly odd gait rather than collapse. A control for 'is it noise or
  is it just being wrong'.

## Reading the result

Watch `style_r/step`. The documented failure fingerprint is style
reward collapsing to ~0 while the discriminator saturates near 100%
accuracy (`CLAUDE.md`, 'AMP failure-mode fingerprints'). Compare each
variant against `jitter_0deg`, which is the uncorrupted control and
should reproduce whatever the current baseline does.

Keep every other knob fixed across runs -- same seed, same iterations,
same env cfg. The point is a curve, not a best result.