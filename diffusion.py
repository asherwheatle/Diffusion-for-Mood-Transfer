"""Gaussian diffusion with v-prediction objective and cosine noise schedule."""

import math
import torch


def cosine_beta_schedule(num_timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = torch.linspace(0, num_timesteps, num_timesteps + 1)
    alpha_bar = torch.cos(((steps / num_timesteps) + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clamp(betas, 0.0001, 0.999)


class GaussianDiffusion:
    """
    Gaussian diffusion with v-prediction objective (Salimans & Ho, 2022).

    Paper section IV-C specifies:
      - v-objective
      - DPM-Solver++ sampler (we use DDIM for simplicity)
      - CFG scale 7 applied to text only (not melody)
    """

    def __init__(self, num_timesteps: int = 1000, device: str = "cpu"):
        self.num_timesteps = num_timesteps
        betas = cosine_beta_schedule(num_timesteps).to(device)
        alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(alphas, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)

    def to(self, device):
        self.alpha_bar = self.alpha_bar.to(device)
        self.sqrt_alpha_bar = self.sqrt_alpha_bar.to(device)
        self.sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to(device)
        return self

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        sqrt_omab = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
        return sqrt_ab * x0 + sqrt_omab * noise

    def v_target(self, x0: torch.Tensor, noise: torch.Tensor,
                 t: torch.Tensor) -> torch.Tensor:
        sqrt_ab = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        sqrt_omab = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
        return sqrt_ab * noise - sqrt_omab * x0

    def predict_x0_from_v(self, x_t: torch.Tensor, v: torch.Tensor,
                          t: torch.Tensor) -> torch.Tensor:
        sqrt_ab = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        sqrt_omab = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
        return sqrt_ab * x_t - sqrt_omab * v

    def predict_eps_from_v(self, x_t: torch.Tensor, v: torch.Tensor,
                           t: torch.Tensor) -> torch.Tensor:
        sqrt_ab = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        sqrt_omab = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
        return sqrt_omab * x_t + sqrt_ab * v

    def ddim_step(self, x_t: torch.Tensor, v: torch.Tensor,
                  t: torch.Tensor, t_prev: torch.Tensor,
                  eta: float = 0.0) -> torch.Tensor:
        """One DDIM step from t to t_prev.

        eta=0 (default) is the deterministic DDIM update. eta>0 injects the
        ancestral noise term, which matters for SDEdit: a fully deterministic
        trajectory started from a lightly-noised input tends to retrace that
        input, leaving the conditioning little room to move it. eta=1 recovers
        DDPM-style ancestral sampling.
        """
        x0_pred = self.predict_x0_from_v(x_t, v, t)
        eps_pred = self.predict_eps_from_v(x_t, v, t)
        ab = self.alpha_bar[t].view(-1, 1, 1, 1)
        ab_prev = self.alpha_bar[t_prev].view(-1, 1, 1, 1)
        sqrt_ab_prev = self.sqrt_alpha_bar[t_prev].view(-1, 1, 1, 1)

        if eta == 0.0:
            sqrt_omab_prev = self.sqrt_one_minus_alpha_bar[t_prev].view(-1, 1, 1, 1)
            return sqrt_ab_prev * x0_pred + sqrt_omab_prev * eps_pred

        sigma = eta * torch.sqrt(torch.clamp(
            (1 - ab_prev) / (1 - ab) * (1 - ab / ab_prev), min=0.0))
        dir_coef = torch.sqrt(torch.clamp(1 - ab_prev - sigma ** 2, min=0.0))
        x_prev = sqrt_ab_prev * x0_pred + dir_coef * eps_pred
        return x_prev + sigma * torch.randn_like(x_t)
