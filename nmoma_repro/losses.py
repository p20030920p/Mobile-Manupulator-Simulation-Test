from __future__ import annotations

import numpy as np


def reconstruction_loss(predicted: np.ndarray, target: np.ndarray) -> float:
    predicted = np.asarray(predicted, dtype=float)
    target = np.asarray(target, dtype=float)
    if predicted.shape != target.shape:
        raise ValueError("predicted and target must have the same shape")
    return float(np.mean((predicted - target) ** 2))


def smoothness_loss(path: np.ndarray) -> float:
    values = np.asarray(path, dtype=float)
    if values.ndim != 2:
        raise ValueError("path must have shape (path_length, path_dim)")
    if len(values) < 3:
        return 0.0
    acceleration = values[2:] - 2.0 * values[1:-1] + values[:-2]
    return float(np.mean(np.linalg.norm(acceleration, axis=1)))


def uniform_loss(path: np.ndarray) -> float:
    values = np.asarray(path, dtype=float)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("path must have shape (path_length, path_dim)")
    segment_lengths = np.linalg.norm(np.diff(values[:, :2], axis=0), axis=1)
    mean_length = float(np.mean(segment_lengths))
    if mean_length <= 1e-12:
        return 0.0
    return float(np.std(segment_lengths) / mean_length)


def safety_loss(path: np.ndarray, sdf_values: np.ndarray, threshold: float = 0.05) -> float:
    path = np.asarray(path, dtype=float)
    sdf_values = np.asarray(sdf_values, dtype=float)
    if sdf_values.shape[0] != path.shape[0]:
        raise ValueError("sdf_values must have one value per path state")
    return float(np.mean(np.maximum(threshold - sdf_values, 0.0)))
