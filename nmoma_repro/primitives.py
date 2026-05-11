from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PrimitiveLibrary:
    primitives: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.primitives, dtype=float)
        if values.ndim != 3:
            raise ValueError("primitives must have shape (primitive_count, path_length, path_dim)")
        object.__setattr__(self, "primitives", values)

    def nearest(self, path: np.ndarray) -> int:
        target = np.asarray(path, dtype=float).reshape(1, -1)
        candidates = self.primitives.reshape(self.primitives.shape[0], -1)
        distances = np.linalg.norm(candidates - target, axis=1)
        return int(np.argmin(distances))

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, primitives=self.primitives)

    @staticmethod
    def load(path: str | Path) -> "PrimitiveLibrary":
        data = np.load(Path(path))
        return PrimitiveLibrary(data["primitives"])


def build_primitive_library(paths: np.ndarray, primitive_count: int = 32, seed: int = 0, iterations: int = 25) -> PrimitiveLibrary:
    data = np.asarray(paths, dtype=float)
    if data.ndim != 3:
        raise ValueError("paths must have shape (sample_count, path_length, path_dim)")
    if primitive_count < 1:
        raise ValueError("primitive_count must be positive")

    rng = np.random.default_rng(seed)
    flat = data.reshape(data.shape[0], -1)
    effective_count = min(primitive_count, data.shape[0])
    centers = flat[rng.choice(data.shape[0], size=effective_count, replace=False)].copy()

    for _ in range(iterations):
        distances = np.linalg.norm(flat[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)
        for cluster_index in range(effective_count):
            members = flat[labels == cluster_index]
            if len(members) > 0:
                centers[cluster_index] = members.mean(axis=0)

    primitives = centers.reshape(effective_count, data.shape[1], data.shape[2])
    if effective_count < primitive_count:
        pad = np.repeat(primitives[-1:, :, :], primitive_count - effective_count, axis=0)
        primitives = np.concatenate([primitives, pad], axis=0)
    return PrimitiveLibrary(primitives)
