"""Launch amp-rsl-rl's AMPOnPolicyRunner on the deer walk task.

Unlike ``training/scripts/train.py``, this does NOT defer to Isaac Lab's
stock rsl_rl trainer -- it builds the env + runner directly so we can wire
``AMP_PPO`` and the ``amp`` observation group.

Usage (PowerShell, from Isaac Lab root):

    $env:PYTHONPATH = 'C:\\path\\to\\aniML_world'
    $env:ISAACLAB_ROOT = 'C:\\isaac-lab\\IsaacLab'
    .\\isaaclab.bat -p C:\\path\\to\\aniML_world\\training\\scripts\\train_amp.py `
        --task AniML-Deer-USD-Walk-AMP-v0 --num_envs 1024 --headless

Args:
    --task: Gym task ID (default: AniML-Deer-USD-Walk-AMP-v0)
    --num_envs: number of parallel envs (default: 1024)
    --max_iterations: override runner's max_iterations from config
    --headless: no GUI
    --seed: override runner seed
    --log_root: base directory for logs (default: logs/rsl_rl/<experiment_name>)

Smoke test recipe:
    --num_envs 64 --max_iterations 10 --headless

Note on dependencies:
    - amp-rsl-rl 1.2.0
    - rsl-rl-lib 3.0.0 (not 4.x/5.x -- amp-rsl-rl imports ``ActorCritic``
      which was removed after 3.x).
    - Isaac Lab's bundled rsl_rl may collide with this. If Isaac Lab vendors
      its own rsl_rl, set up a separate venv with the pinned versions and
      ensure it takes precedence.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_path() -> None:
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def main() -> None:
    _ensure_path()

    # Isaac Lab's AppLauncher parses its own CLI args, so we parse ours first
    # then pass the residue through.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task", default="AniML-Deer-USD-Walk-AMP-v0")
    parser.add_argument("--num_envs", type=int, default=2500)
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--task_reward_ratio",
        type=float,
        default=None,
        help=(
            "Override AMPOnPolicyRunner's hardcoded 50/50 blend of task vs "
            "style reward. If set to R, reward = R*task + (1-R)*style. Default "
            "None keeps the library's 0.5/0.5 blend. Useful values: 0.8 means "
            "task dominates, 0.2 means AMP style dominates."
        ),
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to a .pt checkpoint to warm-start from (calls runner.load()).",
    )
    parser.add_argument(
        "--log_root",
        default=str(_repo_root() / "logs" / "rsl_rl"),
        help="Root directory for logs. Default: <repo>/logs/rsl_rl.",
    )
    # NOTE: Do NOT add --device here -- AppLauncher.add_app_launcher_args
    # adds that. Use args.device (from AppLauncher) after parse.

    # --- Isaac Lab AppLauncher ---
    # Need to launch the simulator before importing isaaclab modules.
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args, extras = parser.parse_known_args()
    # Pass remaining args (e.g. --headless) to AppLauncher.
    sys.argv = [sys.argv[0]] + extras
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # --- Now safe to import Isaac Lab symbols ---
    import gymnasium as gym
    import torch

    import training.tasks  # noqa: F401 -- registers AniML-* gym tasks
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg

    from amp_rsl_rl.runners import AMPOnPolicyRunner

    # --- Resolve configs ---
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)

    # Load the AMP runner cfg dict from the registration entry point.
    task_spec = gym.spec(args.task)
    amp_cfg_ep = task_spec.kwargs.get("amp_rsl_rl_cfg_entry_point")
    if amp_cfg_ep is None:
        raise RuntimeError(
            f"Task {args.task!r} has no 'amp_rsl_rl_cfg_entry_point' in its "
            "gym registration. Use an AMP-enabled task."
        )
    module_name, attr_name = amp_cfg_ep.split(":")
    import importlib
    mod = importlib.import_module(module_name)
    runner_cfg: dict = dict(getattr(mod, attr_name))  # shallow copy so we can mutate

    if args.max_iterations is not None:
        runner_cfg["max_iterations"] = args.max_iterations
    if args.seed is not None:
        runner_cfg["seed"] = args.seed

    experiment_name = runner_cfg.get("experiment_name", "animl_deer_amp_walk")
    run_name = runner_cfg.get("run_name", "") or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path(args.log_root) / experiment_name / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train_amp] experiment={experiment_name}  run={run_name}")
    print(f"[train_amp] log_dir={log_dir}")
    print(f"[train_amp] num_envs={env_cfg.scene.num_envs}  device={args.device}")
    print(f"[train_amp] dataset path={runner_cfg['dataset']['amp_data_path']}")
    print(f"[train_amp] datasets={runner_cfg['dataset']['datasets']}")

    # --- Build env and runner ---
    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    # amp-rsl-rl expects a rsl_rl VecEnv. Isaac Lab provides the adapter.
    vec_env = RslRlVecEnvWrapper(env.unwrapped)  # type: ignore[arg-type]

    # Seed
    seed = runner_cfg.get("seed", 42)
    torch.manual_seed(seed)

    runner = AMPOnPolicyRunner(
        env=vec_env,
        train_cfg=runner_cfg,
        log_dir=str(log_dir),
        device=args.device,
    )

    # ---- AMP normaliser: EXPERT statistics only (r12) ----------------------
    # amp_ppo.py calls
    #     discriminator.update_normalization(expert_state, expert_next,
    #                                        policy_state, policy_next)
    # i.e. it folds POLICY observations into the running mean/std that normalises the
    # discriminator's input. That is self-defeating: when the policy drifts to a joint
    # angle the expert never reaches (measured in r10: the left hoof ankle at -112 deg
    # vs the reference's -65 deg, on 16 of 22 leg joints), those samples inflate that
    # dimension's variance, both sides get divided by the inflated std, and the
    # NORMALISED gap shrinks. The discriminator loses resolution precisely where the
    # policy misbehaves -- the further it deviates, the less AMP can see it. Policy
    # samples also vastly outnumber the 55 expert frames, so the statistics are
    # policy-dominated. Result: amp_loss sat at ln(2) ("cannot distinguish") while the
    # policy was visibly out of distribution.
    # Fix: normalise by the DEMONSTRATION distribution, as GAIL/AMP implementations
    # normally do, so out-of-distribution policy values stay large and visible.
    # PROBE GATE (2026-07-26): ANIML_DEER_AMP_LEGACY_NORM=1 skips the patch, restoring
    # the pre-r12 (buggy) expert+policy statistics. Exists for ONE reason: r10 -- the
    # last successful from-scratch run -- trained WITHOUT this patch (its log lacks the
    # patch banner), and from-scratch has been broken since. Hypothesised mechanism:
    # expert-only stats make a random policy maximally out-of-distribution, the
    # discriminator saturates instantly, style reward flatlines at ~0, and the 0.5*style
    # half of the reward carries no gradient during exactly the phase that needs it.
    # The buggy normaliser accidentally SOFTENED the discriminator for random policies.
    # Diagnostic only -- do not ship runs with this set; if confirmed, the real fix is
    # annealing (start legacy / blend toward expert-only), not keeping the bug.
    _disc = runner.alg.discriminator
    _legacy_norm = os.environ.get("ANIML_DEER_AMP_LEGACY_NORM", "0") == "1"
    if _legacy_norm:
        print("[train_amp] PROBE: legacy expert+policy normaliser (patch SKIPPED "
              "via ANIML_DEER_AMP_LEGACY_NORM=1) -- diagnostic, do not ship", flush=True)
    elif getattr(_disc, "empirical_normalization", False):
        _orig_update_norm = _disc.update_normalization

        def _expert_only_update(*batches, _f=_orig_update_norm):
            # amp_ppo passes (expert_state, expert_next, policy_state, policy_next)
            return _f(*batches[:2]) if len(batches) >= 4 else _f(*batches)

        _disc.update_normalization = _expert_only_update
        print("[train_amp] AMP normaliser patched: expert-only statistics "
              "(was expert+policy, which hid the policy's own outliers)", flush=True)
    else:
        print("[train_amp] discriminator has empirical_normalization off; "
              "no normaliser patch needed", flush=True)

    # ---- B / G: hoof height and hoof speed in the AMP observation ----------
    # The env side adds 4 columns per channel (training/polyart/observations); the
    # DATASET side must add the same columns, in the same order, or the discriminator
    # compares mismatched vectors and the run is silently meaningless. The reference
    # clips have no link positions, so they are reconstructed by the same FK the mesh
    # renderer uses. Guarded: if the widths disagree afterwards, refuse to train.
    #
    # ORDER IS HEIGHTS THEN SPEEDS on both sides. The width guard below cannot catch a
    # transposition -- 4+4 is 8 either way -- so the ordering is enforced by convention
    # in exactly two places: AMPObservationsCfg (hoof_height then hoof_speed, both last)
    # and hoof_height_patch.attach. Change one, change the other.
    _hoof_h = os.environ.get("ANIML_DEER_AMP_HOOF_H", "0") == "1"
    _hoof_v = os.environ.get("ANIML_DEER_AMP_HOOF_V", "0") == "1"
    if _hoof_h or _hoof_v:
        from tools.rig_pipeline.hoof_height_patch import attach as _attach_hoof
        from tools.rig_pipeline.hoof_height_patch import resolve_loader as _amp_loader
        _urdf = os.environ.get(
            "ANIML_DEER_URDF",
            str(_repo_root() / "assets/deer_polyart_urdf/deer_polyart.urdf"))
        _n = _attach_hoof(runner, _urdf, height=_hoof_h, speed=_hoof_v)
        _env_w = runner._flatten_amp_obs(env.unwrapped.observation_manager.compute()["amp"]).shape[1]
        # Guard the buffer the DISCRIMINATOR actually reads, not the method.
        # `get_amp_dataset_obs` is called exactly once, at AMPLoader construction, to
        # fill all_obs/all_next_obs; `feed_forward_generator` then samples those buffers
        # forever. The old guard checked the patched METHOD, so it read 86 and passed
        # while the buffers were still 82 -- and B died 15 iterations later inside
        # amp_ppo with "expected 172 but got 164". Check what is used, not what is easy.
        _ldr = _amp_loader(runner)
        _ds_w = int(_ldr.all_obs.shape[1])
        _m_w = int(_ldr.motion_data[0].get_amp_dataset_obs(
            torch.tensor([0], device=runner.device)).shape[1])
        if _env_w != _ds_w or _env_w != _m_w:
            raise SystemExit(
                f"[train_amp] HOOF CHANNEL WIDTH MISMATCH: env AMP obs {_env_w}, "
                f"expert buffer {_ds_w}, dataset method {_m_w} "
                f"(height={_hoof_h}, speed={_hoof_v}). All three must agree "
                f"column-for-column; training on a mismatch produces confident garbage.")
        _ch = "+".join([c for c, on in (("height", _hoof_h), ("speed", _hoof_v)) if on])
        print(f"[train_amp] hoof {_ch} ON: +{_n} cols, AMP obs now {_env_w} "
              f"(discriminator can finally see ground relationship)", flush=True)

    # ---- ASE stage 1: skill-discovery reward + encoder --------------------
    # Layered onto AMP rather than replacing it. amp_on_policy_runner computes
    #     style_rewards = discriminator.predict_reward(amp_obs, next_amp_obs)
    # once per env step, which is the only place with both transition endpoints in
    # hand -- so the MI reward is added there. The encoder UPDATE cannot happen there:
    # that call site runs inside torch.inference_mode(), and tensors born in inference
    # mode can never take part in autograd (a plain .clone() still raises on
    # backward()). So predict_reward only CAPTURES transitions, via the
    # empty_like().copy_() escape hatch, and the update is hung off alg.update(),
    # which the runner calls outside inference mode once per iteration.
    _ase_dim = int(os.environ.get("ANIML_DEER_ASE_DIM", "0"))
    if _ase_dim > 0:
        from training.polyart.ase import ASETrainer

        _env_u = env.unwrapped
        # CAREFUL: Discriminator.input_dim is ALREADY the transition width. The runner
        # builds it as `input_dim=num_amp_obs * 2` ("the concatenation of the current
        # and next observation"), so for our 82-dim AMP obs it is 164, not 82.
        # SkillEncoder takes the SINGLE-state width and concatenates internally, so
        # passing 164 built a 328-wide first layer and fed it 164:
        #   RuntimeError: mat1 and mat2 shapes cannot be multiplied (4096x164 and 328x512)
        _amp_dim = int(_disc.input_dim) // 2
        _ase = ASETrainer(
            obs_dim=_amp_dim, z_dim=_ase_dim, device=runner.device,
            mi_scale=float(os.environ.get("ANIML_DEER_ASE_MI_SCALE", "0.5")),
        )
        runner._ase = _ase
        _orig_predict = _disc.predict_reward
        _orig_update = runner.alg.update

        def _ase_predict_reward(s, s_next, *a, _f=_orig_predict, **kw):
            style = _f(s, s_next, *a, **kw)
            z = getattr(_env_u, "ase_z", None)
            if z is None or z.shape[0] != s.shape[0]:
                return style
            _ase.capture(s, s_next, z)
            return style + _ase.mi_reward(s, s_next, z)

        # C1 (2026-07-28): LOG the encoder. Without this, ASE is behaviourally testable
        # (the z-sweep proves the latent is live) but numerically opaque -- there is no
        # record of whether the encoder converged or barely learned, which is the one
        # number you would actually tune mi_scale against. exp4 ran entirely unmeasured.
        _ase_it = {"n": 0}

        def _ase_update(*a, _f=_orig_update, **kw):
            out = _f(*a, **kw)
            loss = _ase.update()
            _ase_it["n"] += 1
            wr = getattr(runner, "writer", None)
            if wr is not None and loss == loss:  # NaN-safe
                it = _ase_it["n"]
                wr.add_scalar("ASE/encoder_loss", loss, it)
                wr.add_scalar("ASE/mi_reward", _ase.last_mi, it)
            if _ase_it["n"] % 50 == 0:
                print(f"[ase] iter~{_ase_it['n']}  encoder_loss={loss:.4f}  "
                      f"mi_reward={_ase.last_mi:.4f}  (loss->0 = z fully recoverable, "
                      f"loss->1 = encoder cannot tell skills apart)", flush=True)
            return out

        _disc.predict_reward = _ase_predict_reward
        runner.alg.update = _ase_update
        print(f"[train_amp] ASE stage 1 ON: z_dim={_ase_dim}, amp_obs={_amp_dim}, "
              f"mi_scale={_ase.mi_scale} (style + MI, task reward should be ~0)",
              flush=True)

        # ---- ASE diversity objective (task #8) ----------------------------
        # Unlike the MI reward, this is NOT a reward -- it is a term on the POLICY LOSS,
        # so it needs the actor and the policy observations together, inside PPO's
        # update. rsl_rl 5.0.1 assembles and backwards its loss entirely within
        # `PPO.update()` (ppo.py:313 -> :363) with no hook in between, and copying that
        # 150-line method to insert one term is exactly the "modelling on old code" trap.
        #
        # Instead, exploit the fact that a second backward() ACCUMULATES into the same
        # .grad buffers. Per minibatch rsl_rl runs:
        #     zero_grad() -> loss.backward() -> clip_grad_norm_(actor) -> optimizer.step()
        # so wrapping optimizer.step() gives a point where the graph is gone but the
        # gradients are live and the step has not happened. We add the diversity
        # gradient there and RE-CLIP the actor, because our contribution lands after
        # rsl_rl's own clip and would otherwise be the one unbounded gradient in the
        # system. The observations for the current minibatch are captured off the
        # actor's forward -- rsl_rl calls it with stochastic_output=True exactly once
        # per minibatch (ppo.py:256), and only inside update() is grad enabled, which
        # distinguishes it from the rollout.
        _div_coef = float(os.environ.get("ANIML_DEER_ASE_DIVERSITY", "0.0"))
        if _div_coef > 0.0:
            import numpy as _np
            import torch.nn as _nn
            from training.polyart.ase import DiversityRegularizer

            # Locate z inside the concatenated policy observation. Computed, not
            # assumed: ase_latent is currently last, but a term added after it would
            # silently shift the slice onto joint data.
            _om = _env_u.observation_manager
            _terms = list(_om.active_terms["policy"])
            _dims = list(_om.group_obs_term_dim["policy"])
            if "ase_latent" not in _terms:
                raise SystemExit("[ase-diversity] no 'ase_latent' term in the policy "
                                 "observation group -- ASE_DIM must be set.")
            _ti = _terms.index("ase_latent")
            _z0 = int(sum(int(_np.prod(d)) for d in _dims[:_ti]))
            _alg = runner.alg
            _div = DiversityRegularizer(
                actor=_alg.actor, z_start=_z0, z_dim=_ase_dim,
                coef=_div_coef,
                target=float(os.environ.get("ANIML_DEER_ASE_DIVERSITY_TAR", "1.0")),
            )
            runner._ase_div = _div
            _orig_step = _alg.optimizer.step

            # Observations are SAMPLED FROM THE ROLLOUT STORAGE rather than captured off
            # the actor's forward. The first version patched `alg.actor.forward` with a
            # local closure, which worked and trained -- and then died at the first
            # checkpoint:
            #     AttributeError: Can't pickle local object 'main.<locals>._capture_fwd'
            # amp_on_policy_runner.save() calls export_policy_as_onnx, whose
            # _clone_module deep-copies the actor through torch.save, so ANY unpicklable
            # instance attribute on that module is a time bomb that only goes off at the
            # first save -- long after a smoke test that saves nothing would have passed.
            # Storage is never serialised (the runner saves state_dicts only), so this
            # is safe by construction rather than by luck.
            #
            # Using a fresh random draw instead of PPO's exact minibatch is fine and
            # arguably better: the diversity term is a regulariser over the state
            # distribution, not a per-sample correction, so it does not need to be
            # aligned with the surrogate loss. Drawing independently each step also
            # covers more of the rollout.
            # Deliberately NOT wrapped in try/except. A regulariser that silently
            # switches itself off would let F1 run for four hours and be reported as a
            # diversity experiment while having no diversity term in it -- the exact
            # shape of the failures this project has already retracted conclusions over.
            # A crash is loud, the smoke test catches it in minutes, and
            # ASE/diversity_bonus in TensorBoard is the standing proof it stayed on.
            def _step_with_diversity(*a, _f=_orig_step, **kw):
                st = getattr(_alg, "storage", None)
                obs_all = getattr(st, "observations", None) if st is not None else None
                if obs_all is not None:
                    flat = obs_all.flatten(0, 1)
                    n = int(flat.batch_size[0])
                    m = max(1, n // max(int(getattr(_alg, "num_mini_batches", 1)), 1))
                    idx = torch.randint(0, n, (m,), device=flat.device)
                    dl = _div.loss(flat[idx])
                    if dl is not None:
                        dl.backward()
                        # Re-clip: our gradient lands AFTER rsl_rl's own clip
                        # (ppo.py:374) and would otherwise be unbounded.
                        _nn.utils.clip_grad_norm_(_alg.actor.parameters(),
                                                  _alg.max_grad_norm)
                return _f(*a, **kw)

            _alg.optimizer.step = _step_with_diversity

            def _div_log(*a, _f=runner.alg.update, **kw):
                out = _f(*a, **kw)
                wr = getattr(runner, "writer", None)
                if wr is not None and _div.last_loss == _div.last_loss:
                    wr.add_scalar("ASE/diversity_loss", _div.last_loss, _ase_it["n"])
                    wr.add_scalar("ASE/diversity_bonus", _div.last_bonus, _ase_it["n"])
                return out

            runner.alg.update = _div_log
            print(f"[train_amp] ASE diversity ON: coef={_div_coef} "
                  f"target={_div.target} z at policy-obs cols [{_z0}:{_z0+_ase_dim}] "
                  f"(bonus->target means action spread tracks latent spread)",
                  flush=True)

    # Monkey-patch the reward blend if the user overrode it. amp_on_policy_runner
    # hardcodes `rewards = 0.5 * rewards + 0.5 * style_rewards`; this patches it
    # to `R * rewards + (1-R) * style_rewards` by wrapping learn() internals.
    if args.task_reward_ratio is not None:
        ratio = float(args.task_reward_ratio)
        if not (0.0 <= ratio <= 1.0):
            raise ValueError(f"--task_reward_ratio must be in [0,1], got {ratio}")
        print(f"[train_amp] using task_reward_ratio={ratio} "
              f"(task={ratio:.2f}, style={1-ratio:.2f})")
        import amp_rsl_rl.runners.amp_on_policy_runner as _runner_mod
        # The simplest intervention: wrap discriminator.predict_reward so it
        # returns the STYLE reward scaled by (1-ratio)/0.5, and rely on the
        # existing 0.5/0.5 blend to then produce the correct combination.
        # reward = 0.5 * task + 0.5 * (style * (1-ratio)/0.5)
        #        = 0.5 * task + (1-ratio) * style
        # That's NOT ratio*task + (1-ratio)*style. Need to ALSO scale task.
        # Do that via env.step reward monkey patch — harder. Use a cleaner
        # approach: override runner.alg.process_env_step? No, rewards are
        # already blended before that.
        # Easiest: scale the style reward directly. Combined with the cfg
        # discriminator.reward_scale this gives us the effective blend.
        # Resulting blend: 0.5*task + 0.5*scale*D.  We want ratio*task +
        # (1-ratio)*D. So we set `task_scale_multiplier = 2*ratio` (applied
        # by overriding env.step wrapper) and `style_scale = 2*(1-ratio)`
        # (via patching predict_reward return).
        task_scale = 2.0 * ratio
        style_scale = 2.0 * (1.0 - ratio)

        orig_predict_reward = runner.discriminator.predict_reward

        def scaled_predict_reward(*a, **kw):
            return orig_predict_reward(*a, **kw) * style_scale
        runner.discriminator.predict_reward = scaled_predict_reward

        # For task: wrap vec_env.step so reward out *= task_scale.
        orig_step = vec_env.step

        def scaled_step(actions):
            obs, rewards, dones, extras = orig_step(actions)
            return obs, rewards * task_scale, dones, extras
        vec_env.step = scaled_step

    if args.resume_from is not None:
        resume_path = Path(args.resume_from)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume ckpt not found: {resume_path}")
        print(f"[train_amp] warm-starting from {resume_path}")
        runner.load(str(resume_path))

    # ---- Defensive std clamp ----
    # r16/r18 both crashed with `RuntimeError: normal expects all elements of
    # std >= 0.0` after a reward overflow drove the PPO update to produce
    # non-finite gradients, which made the actor's std_param go negative.
    # Hot-patch update() so right after each PPO step we clamp std_param to
    # a positive minimum and replace any non-finite values. Combined with
    # the V11 env's zeroed L2 penalties this should make training robust
    # against transient PhysX/reward spikes.
    _STD_FLOOR = 0.05
    _orig_update = runner.alg.update

    def _safe_update(*a, **kw):
        result = _orig_update(*a, **kw)
        ac = runner.alg.actor_critic
        if hasattr(ac, "std") and ac.std is not None:
            with torch.no_grad():
                std = ac.std
                bad = ~torch.isfinite(std)
                if bool(bad.any()):
                    print(f"[train_amp] WARN: non-finite std elements detected, "
                          f"replacing with {_STD_FLOOR}", flush=True)
                    std[bad] = _STD_FLOOR
                std.clamp_(min=_STD_FLOOR)
        # Same for std_param if exposed under that name
        for attr in ("std_param", "log_std", "logstd"):
            p = getattr(ac, attr, None)
            if p is None:
                continue
            with torch.no_grad():
                bad = ~torch.isfinite(p)
                if bool(bad.any()):
                    p[bad] = 0.0  # log_std=0 -> std=1
                if attr == "std_param":
                    p.clamp_(min=_STD_FLOOR)
        return result

    runner.alg.update = _safe_update
    print(f"[train_amp] installed std-clamp safeguard (floor={_STD_FLOOR})",
          flush=True)

    runner.learn(num_learning_iterations=runner_cfg["max_iterations"])

    # Close cleanly
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
