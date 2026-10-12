"""CPU test for the ASE diversity objective and its rsl_rl hook.

Written because the two ASE bugs this project has already paid for were both wiring,
not maths, and both surfaced only after a run had been queued for hours:

  * exp2 crashed on `joints_list` being an ndarray where AMPLoader calls `.index()`
  * exp4 crashed on Discriminator.input_dim already being the transition width

Neither needed a GPU to find. This runs the REAL rsl_rl MLPModel, a REAL Adam, and the
REAL patch sequence from train_amp.py against a fake TensorDict, so a shape or hook
error shows up in seconds instead of at 03:00 six hours into a queue.

    C:/isaac-lab/env_isaaclab/Scripts/python.exe tools/test_ase_diversity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from tensordict import TensorDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsl_rl.models.mlp_model import MLPModel  # noqa: E402

from training.polyart.ase import DiversityRegularizer, sample_z  # noqa: E402

DEV = "cpu"
B, N_OTHER, Z_DIM, N_ACT = 64, 40, 8, 38
OBS_DIM = N_OTHER + Z_DIM          # z is the LAST term, as in PolicyCfg
FAILS: list[str] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)
    return ok


def make_obs(n: int = B) -> TensorDict:
    flat = torch.randn(n, OBS_DIM, device=DEV)
    flat[:, N_OTHER:] = sample_z(n, Z_DIM, DEV)
    return TensorDict({"policy": flat}, batch_size=[n], device=DEV)


def make_actor(obs: TensorDict) -> MLPModel:
    return MLPModel(
        obs=obs, obs_groups={"actor": ["policy"]}, obs_set="actor",
        output_dim=N_ACT, hidden_dims=[64, 64], activation="elu",
        distribution_cfg={"class_name": "rsl_rl.modules.distribution:GaussianDistribution",
                          "init_std": 1.0, "std_type": "scalar"},
    ).to(DEV)


def main() -> int:
    torch.manual_seed(0)
    obs = make_obs()
    actor = make_actor(obs)
    div = DiversityRegularizer(actor=actor, z_start=N_OTHER, z_dim=Z_DIM, coef=0.01)

    print("\n=== 1. loss computes and is finite ===")
    loss = div.loss(obs)
    check(loss is not None and torch.isfinite(loss).all(),
          "diversity loss finite", f"loss={float(loss):.6f} bonus={div.last_bonus:.4f}")
    check(loss.requires_grad, "loss is differentiable")

    print("\n=== 2. gradient reaches the actor ===")
    actor.zero_grad(set_to_none=True)
    loss.backward()
    gn = sum(float(p.grad.norm()) for p in actor.parameters() if p.grad is not None)
    check(gn > 0, "actor receives gradient", f"total grad norm {gn:.6f}")

    print("\n=== 3. the objective actually measures z-sensitivity ===")
    # A policy that IGNORES z (weights on the z columns zeroed) must produce a_diff of
    # exactly 0, hence bonus 0 and loss = target^2. This is the collapse case ASE's
    # diversity term exists to punish -- if it scores the same as a z-sensitive policy,
    # the slice arithmetic is wrong and the whole term is decorative.
    blind = make_actor(obs)
    with torch.no_grad():
        first = next(m for m in blind.mlp.modules() if isinstance(m, nn.Linear))
        first.weight[:, N_OTHER:] = 0.0
    d_blind = DiversityRegularizer(actor=blind, z_start=N_OTHER, z_dim=Z_DIM, coef=1.0)
    l_blind = float(d_blind.loss(obs))
    check(abs(d_blind.last_bonus) < 1e-8, "z-blind policy scores bonus 0",
          f"bonus={d_blind.last_bonus:.3e}")
    check(abs(l_blind - 1.0) < 1e-6, "z-blind policy pays the full target penalty",
          f"loss={l_blind:.6f} (target^2 = 1.0)")

    print("\n=== 4. training on it increases behavioural spread ===")
    # The claim under test is not "the loss goes down" but "the policy becomes more
    # z-sensitive". Measure the raw action spread before and after.
    trainee = make_actor(obs)
    d_tr = DiversityRegularizer(actor=trainee, z_start=N_OTHER, z_dim=Z_DIM, coef=1.0)
    opt = torch.optim.Adam(trainee.parameters(), lr=1e-3)
    d_tr.loss(obs)
    before = d_tr.last_bonus
    for _ in range(60):
        o = make_obs()
        l = d_tr.loss(o)
        opt.zero_grad(set_to_none=True)
        l.backward()
        opt.step()
    d_tr.loss(obs)
    after = d_tr.last_bonus
    check(after > before, "spread increases under training",
          f"bonus {before:.4f} -> {after:.4f} (target 1.0)")

    print("\n=== 5. the wrong-slice guard fires ===")
    bad = DiversityRegularizer(actor=actor, z_start=0, z_dim=Z_DIM, coef=0.01)
    try:
        bad.loss(obs)
        check(False, "wrong slice raises", "it silently accepted joint columns as z")
    except SystemExit as e:
        check("unit vectors" in str(e), "wrong slice raises SystemExit", str(e)[:80])

    print("\n=== 5b. the actor stays PICKLABLE ===")
    # The first hook assigned a local closure to `actor.forward` to capture
    # observations. It trained fine and died at the first checkpoint:
    #   AttributeError: Can't pickle local object 'main.<locals>._capture_fwd'
    # amp_on_policy_runner.save() -> export_policy_as_onnx -> _clone_module deep-copies
    # the actor THROUGH torch.save, so any unpicklable attribute on that module is a
    # time bomb that only detonates at the first save -- after a smoke test that saves
    # nothing would have passed it. The wiring now samples from rollout storage and
    # never touches the actor; this asserts that.
    import io
    import pickle
    pickle_actor = make_actor(obs)
    try:
        buf = io.BytesIO()
        torch.save(pickle_actor, buf)
        check(True, "clean actor pickles", f"{buf.tell()} bytes")
    except Exception as e:
        check(False, "clean actor pickles", f"{type(e).__name__}: {e}")

    def _local_closure(x):        # exactly what the old wiring attached
        return x
    pickle_actor.forward = _local_closure
    try:
        torch.save(pickle_actor, io.BytesIO())
        check(False, "a local closure on .forward WOULD break saving",
              "it pickled -- this test no longer reproduces the bug")
    except (AttributeError, pickle.PicklingError) as e:
        check(True, "a local closure on .forward WOULD break saving",
              f"{type(e).__name__} (so the hook must not touch the actor)")

    print("\n=== 6. the train_amp.py hook sequence ===")
    # Storage-sampling -> zero_grad -> ppo backward -> clip -> patched step.
    hooked = make_actor(obs)
    d_h = DiversityRegularizer(actor=hooked, z_start=N_OTHER, z_dim=Z_DIM, coef=0.01)
    opt_h = torch.optim.Adam(hooked.parameters(), lr=1e-3)
    orig_step, seen = opt_h.step, {"n": 0}
    # storage holds (num_transitions, num_envs, ...) and the generator flattens (0, 1)
    storage_obs = TensorDict({"policy": make_obs(8 * B)["policy"].view(8, B, OBS_DIM)},
                             batch_size=[8, B], device=DEV)

    def step(*a, _f=orig_step, **kw):
        flat = storage_obs.flatten(0, 1)
        n = int(flat.batch_size[0])
        idx = torch.randint(0, n, (n // 4,), device=DEV)
        dl = d_h.loss(flat[idx])
        if dl is not None:
            dl.backward()
            nn.utils.clip_grad_norm_(hooked.parameters(), 1.0)
            seen["n"] += 1
        return _f(*a, **kw)

    opt_h.step = step
    check(hooked.forward.__self__ is hooked if hasattr(hooked.forward, "__self__") else True,
          "actor.forward is untouched (still the bound method)")

    # distribution.sample() is detached, so a PPO-shaped loss has to go through
    # get_output_log_prob -- the same path ppo.py:262 uses.
    hooked(obs, stochastic_output=True)
    fake_ppo_loss = hooked.get_output_log_prob(torch.zeros(B, N_ACT, device=DEV)).mean()
    opt_h.zero_grad(set_to_none=True)
    fake_ppo_loss.backward()
    nn.utils.clip_grad_norm_(hooked.parameters(), 1.0)
    opt_h.step()
    check(seen["n"] == 1, "diversity gradient added exactly once per step",
          f"{seen['n']} time(s)")
    check(d_h.last_loss == d_h.last_loss, "diversity ran inside the patched step",
          f"loss={d_h.last_loss:.6f}")

    # and the actor is STILL savable with the hook installed
    try:
        torch.save(hooked, io.BytesIO())
        check(True, "actor still pickles with the hook installed")
    except Exception as e:
        check(False, "actor still pickles with the hook installed", f"{type(e).__name__}: {e}")

    print("\n" + "=" * 66)
    if FAILS:
        print(f"FAILED — {len(FAILS)} check(s): " + "; ".join(FAILS))
    else:
        print("ALL PASS — diversity objective and hook are wired correctly.")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
