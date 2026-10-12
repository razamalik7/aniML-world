"""ASE stage 1 — adversarial skill embeddings, layered onto the existing AMP setup.

WHAT ASE IS. Stage 1 trains a low-level policy pi(a | s, z) where z is a random unit
vector drawn per episode: a dial with no labels, mapping each setting to a different
way of moving. There is NO task reward. Two signals:

  * STYLE  — the existing AMP discriminator: "does this look like the dataset?"
  * SKILL  — an encoder q(z | s, s') tries to recover WHICH z produced a transition,
             and the policy is rewarded when it succeeds.
  * SPREAD — a diversity regulariser on the policy loss, requiring the ACTIONS for two
             latents to differ in proportion to the latents' own distance.

The encoder is the whole point. With style alone, the policy simply IGNORES z: the
discriminator only cares that motion looks real, not which dial setting produced it,
so every z collapses to the same walk. To let the encoder recover z, the policy is
forced to make z=A visibly different from z=B. That pressure is what spreads the
dataset across the latent space instead of collapsing it.

The encoder alone is not enough, though, and we have the measurement to prove it:
exp4 reached MI 0.82 while its z-sweep separated skills by only 22-36 degrees of pose.
"Recoverable" and "different" are not the same requirement -- a fingerprint the encoder
can read costs the policy almost nothing. The diversity objective is what converts
detectability into visible behavioural range; see ``DiversityRegularizer``.

Stage 2 (not here) freezes this policy and trains a small high-level policy that emits
z from task state. Task rewards live there, one per task, all sharing this low level.

IMPLEMENTATION. z lives on the env as ``env.ase_z`` so an ObsTerm can expose it to the
actor and an EventTerm can resample it on reset -- both idiomatic Isaac Lab, and it
keeps the actor's input dimension correct at construction time. The MI reward and the
encoder update are patched onto the amp-rsl-rl runner in train_amp.py: the reward via
``discriminator.predict_reward`` (already the monkey-patch point for the style blend),
the encoder update via ``alg.update``, which runs OUTSIDE inference_mode.

The inference_mode detail is load-bearing. ``predict_reward`` is called inside
``torch.inference_mode()``, and tensors created there are *inference tensors* that can
never participate in autograd -- a plain ``.clone()`` still fails on ``.backward()``.
Capturing them for the encoder update requires the escape hatch used in ``capture``
below, which was verified before this module was written.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def sample_z(n: int, dim: int, device) -> torch.Tensor:
    """n unit vectors, uniform on the (dim-1)-sphere. ASE's latent prior."""
    z = torch.randn(n, dim, device=device)
    return z / z.norm(dim=-1, keepdim=True).clamp(min=1e-8)


class DiversityRegularizer:
    """ASE's SECOND objective -- the one exp4/C2/C3 were missing.

    The encoder reward asks "can q recover z from the transition?". That is satisfiable
    CHEAPLY: the policy only has to make each z leave a detectable fingerprint, and a
    barely-perceptible one is worth as much as a whole new gait. exp4 is the evidence --
    MI 0.82 while its z-sweep separated the skills by only 22-36 degrees of pose. High
    mutual information, near-identical motion. That is not a tuning failure; the
    objective simply does not ask for behavioural spread.

    The diversity objective asks for it directly. For the same state, run the actor
    under the batch's own z and under a freshly drawn z, and require the resulting
    action means to differ IN PROPORTION to how far apart the two latents are:

        a_diff = mean_j ( clamp(mu_z1)_j - clamp(mu_z2)_j )^2      behavioural distance
        z_diff = 0.5 - 0.5 * <z1, z2>          in [0, 1] for unit z; 0 = same skill
        loss   = ( target - a_diff / (z_diff + eps) )^2

    Two properties worth noting. It is a RATIO, so near-identical latents are expected
    to act near-identically and are not punished for it -- only the slope is pinned.
    And it is squared around a target rather than maximised, so it cannot be won by
    flailing: overshooting the target is penalised exactly as much as collapsing.
    That matters here, because every reward this project has bolted on with a one-sided
    "more is better" shape (feet_air_time, joint_ref_range) got gamed within 500 iters.

    Formulation and defaults follow the reference implementation (nv-tlabs/ASE,
    `ase/learning/ase_agent.py`): coefficient 0.01, target 1.0, actions clamped to
    [-1, 1] before comparison. The clamp is load-bearing -- without it the cheapest way
    to raise a_diff is to drive one unbounded action dimension to infinity.
    """

    def __init__(self, actor, z_start: int, z_dim: int, obs_group: str = "policy",
                 coef: float = 0.01, target: float = 1.0, action_clamp: float = 1.0):
        self.actor = actor
        self.sl = slice(z_start, z_start + z_dim)
        self.z_dim = z_dim
        self.obs_group = obs_group
        self.coef = coef
        self.target = target
        self.action_clamp = action_clamp
        self.last_loss = float("nan")
        self.last_bonus = float("nan")
        self._checked = False

    def _verify_slice(self, z_old: torch.Tensor) -> None:
        """Refuse to train on the wrong columns.

        The latent slice is computed from the observation manager's term widths. If a
        term is ever reordered or resized, that arithmetic silently starts pointing at
        joint velocities, and the diversity loss would still run, still log a plausible
        number, and quietly optimise garbage -- the exact failure shape that cost this
        project r7-r9 (clamped RSI spawns) and exp1 (turn-contaminated expert data).
        z is drawn on the unit sphere, so ||z|| == 1 is a free, decisive check.
        """
        n = z_old.norm(dim=-1)
        if not bool(((n - 1.0).abs() < 1e-3).all()):
            raise SystemExit(
                f"[ase-diversity] latent slice {self.sl} does not hold unit vectors "
                f"(norms min={float(n.min()):.4f} max={float(n.max()):.4f}). The "
                f"observation layout changed; recompute the ase_latent offset."
            )
        self._checked = True

    def loss(self, obs) -> torch.Tensor | None:
        """Diversity loss for one minibatch, or None if the latent is not present."""
        flat = obs[self.obs_group]
        if flat.shape[-1] < self.sl.stop:
            return None
        z_old = flat[:, self.sl]
        if not self._checked:
            self._verify_slice(z_old)

        z_new = sample_z(flat.shape[0], self.z_dim, flat.device)
        obs_new = obs.clone()
        flat_new = flat.clone()
        flat_new[:, self.sl] = z_new
        obs_new[self.obs_group] = flat_new

        # stochastic_output=False returns the distribution MEAN, which is what the
        # diversity objective compares -- sampled actions would swamp the signal with
        # exploration noise.
        mu_old = self.actor(obs)
        mu_new = self.actor(obs_new)

        c = self.action_clamp
        a_diff = (mu_old.clamp(-c, c) - mu_new.clamp(-c, c)).square().mean(dim=-1)
        z_diff = 0.5 - 0.5 * (z_old * z_new).sum(dim=-1)
        bonus = a_diff / (z_diff + 1e-5)
        loss = (self.target - bonus).square().mean()

        self.last_bonus = float(bonus.mean())
        self.last_loss = float(loss)
        return self.coef * loss


class SkillEncoder(nn.Module):
    """q(z | s, s') — recovers which latent produced a transition.

    Outputs a unit vector (the mean of a von Mises-Fisher on the same hypersphere the
    latent is drawn from), so similarity to the true z is just a dot product.
    """

    def __init__(self, obs_dim: int, z_dim: int, hidden=(512, 256)):
        super().__init__()
        layers, d = [], obs_dim * 2
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ELU()]
            d = h
        layers += [nn.Linear(d, z_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
        mu = self.net(torch.cat([s, s_next], dim=-1))
        return mu / mu.norm(dim=-1, keepdim=True).clamp(min=1e-8)


class ASETrainer:
    """Owns the encoder, the MI reward, and a small replay of captured transitions."""

    def __init__(self, obs_dim: int, z_dim: int, device, lr: float = 1e-4,
                 mi_scale: float = 0.5, capacity: int = 64):
        self.encoder = SkillEncoder(obs_dim, z_dim).to(device)
        self.opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.device = device
        self.z_dim = z_dim
        self.mi_scale = mi_scale
        self.capacity = capacity
        self.buf: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.last_loss = float("nan")
        self.last_mi = float("nan")

    @torch.no_grad()
    def mi_reward(self, s: torch.Tensor, s_next: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Reward = cosine similarity between the recovered and the true latent.

        Clamped at 0 so a wrong guess is merely unrewarded, never punished -- the same
        correction that had to be applied to feet_air_time, where punishing a failed
        attempt made doing nothing strictly better than trying.
        """
        mu = self.encoder(s, s_next)
        r = (mu * z).sum(dim=-1).clamp(min=0.0)
        self.last_mi = float(r.mean())
        return r * self.mi_scale

    def capture(self, s: torch.Tensor, s_next: torch.Tensor, z: torch.Tensor) -> None:
        """Store a transition for the encoder update.

        ``predict_reward`` runs inside torch.inference_mode(); tensors born there are
        inference tensors and raise RuntimeError on backward() even after .clone().
        empty_like().copy_() inside inference_mode(False) produces a normal tensor.
        """
        with torch.inference_mode(False):
            self.buf.append((
                torch.empty_like(s).copy_(s),
                torch.empty_like(s_next).copy_(s_next),
                torch.empty_like(z).copy_(z),
            ))
        if len(self.buf) > self.capacity:
            self.buf.pop(0)

    def update(self, epochs: int = 1) -> float:
        """Train the encoder to recover z. Loss = 1 - cosine similarity."""
        if not self.buf:
            return float("nan")
        total, n = 0.0, 0
        for _ in range(epochs):
            for s, s_next, z in self.buf:
                mu = self.encoder(s, s_next)
                loss = (1.0 - (mu * z).sum(dim=-1)).mean()
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                self.opt.step()
                total += float(loss); n += 1
        self.buf.clear()
        self.last_loss = total / max(n, 1)
        return self.last_loss


# ---------------------------------------------------------------------------
# Isaac Lab glue: the latent lives on the env so an ObsTerm can expose it to the
# actor and an EventTerm can resample it on reset.
# ---------------------------------------------------------------------------
def _ensure_buf(env, dim: int) -> torch.Tensor:
    buf = getattr(env, "ase_z", None)
    if buf is None or buf.shape[0] != env.num_envs or buf.shape[1] != dim:
        buf = sample_z(env.num_envs, dim, env.device)
        env.ase_z = buf
    return buf


def observe_latent(env, dim: int = 8) -> torch.Tensor:
    """ObsTerm: the actor must SEE z, otherwise it cannot condition on it."""
    return _ensure_buf(env, dim)


def resample_latent(env, env_ids, dim: int = 8) -> None:
    """EventTerm (mode='reset'): draw a fresh skill for each restarting episode.

    Per-episode resampling is what makes z a *skill* dial rather than a constant: the
    policy sees many z values against the same dynamics, and the encoder's job is only
    solvable if different z produce visibly different motion.
    """
    buf = _ensure_buf(env, dim)
    if env_ids is None or (hasattr(env_ids, "numel") and env_ids.numel() == env.num_envs):
        buf.copy_(sample_z(env.num_envs, dim, env.device))
    else:
        buf[env_ids] = sample_z(len(env_ids), dim, env.device)
