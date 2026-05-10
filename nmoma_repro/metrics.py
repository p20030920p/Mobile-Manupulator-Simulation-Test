from __future__ import annotations

import numpy as np


def path_length(path: np.ndarray) -> float:
    values = np.asarray(path, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("path must have shape (path_length, path_dim)")
    deltas = np.diff(values[:, :2], axis=0)
    return float(np.sum(np.linalg.norm(deltas, axis=1)))


def trajectory_duration(path: np.ndarray, nominal_speed: float = 0.5) -> float:
    if nominal_speed <= 0.0:
        raise ValueError("nominal_speed must be positive")
    return path_length(path) / nominal_speed


def diversity_score(paths: np.ndarray, position_bins: int = 24) -> float:
    values = np.asarray(paths, dtype=float)
    if values.ndim != 3:
        raise ValueError("paths must have shape (sample_count, path_length, path_dim)")
    if values.shape[0] <= 1:
        return 0.0

    positions = values[:, :, :2]
    lo = positions.min(axis=(0, 1))
    hi = positions.max(axis=(0, 1))
    span = np.maximum(hi - lo, 1e-9)
    occupancy = []
    for candidate in positions:
        cells = np.floor((candidate - lo) / span * (position_bins - 1)).astype(int)
        occupancy.append({(int(x), int(y)) for x, y in cells})

    union = set().union(*occupancy)
    if not union:
        return 0.0
    ious = []
    for cells in occupancy:
        ious.append(len(cells & union) / len(cells | union))
    return float(max(0.0, 1.0 - np.mean(ious)))
