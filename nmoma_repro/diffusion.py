from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DiffusionSchedule:
    betas: np.ndarray
    alphas: np.ndarray
    alpha_bars: np.ndarray

    @staticmethod
    def linear(total_steps: int = 1200, beta_start: float = 1e-4, beta_end: float = 2e-2) -> "DiffusionSchedule":
        if total_steps < 1:
            raise ValueError("total_steps must be positive")
        betas = np.linspace(beta_start, beta_end, total_steps, dtype=float)
        alphas = 1.0 - betas
        alpha_bars = np.cumprod(alphas)
        return DiffusionSchedule(betas=betas, alphas=alphas, alpha_bars=alpha_bars)


def diffuse_from_primitive(
    primitive: np.ndarray,
    schedule: DiffusionSchedule,
    truncated_step: int,
    noise: np.ndarray | None = None,
    seed: int | None = None,
) -> np.ndarray:
    primitive = np.asarray(primitive, dtype=float)
    if not 1 <= truncated_step <= len(schedule.alpha_bars):
        raise ValueError("truncated_step must be within the schedule")
    if noise is None:
        rng = np.random.default_rng(seed)
        noise = rng.normal(size=primitive.shape)
    noise = np.asarray(noise, dtype=float)
    if noise.shape != primitive.shape:
        raise ValueError("noise must match primitive shape")

    alpha_bar = schedule.alpha_bars[truncated_step - 1]
    return np.sqrt(alpha_bar) * primitive + np.sqrt(1.0 - alpha_bar) * noise
