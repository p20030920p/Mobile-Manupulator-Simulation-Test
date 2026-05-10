from __future__ import annotations

import numpy as np


def prune_path(path: np.ndarray, tolerance: float = 1e-4) -> np.ndarray:
    values = np.asarray(path, dtype=float)
    if values.ndim != 2:
        raise ValueError("path must have shape (path_length, path_dim)")
    if len(values) <= 2:
        return values.copy()

    kept = [values[0]]
    for index in range(1, len(values) - 1):
        prev_xy = kept[-1][:2]
        curr_xy = values[index][:2]
        next_xy = values[index + 1][:2]
        if np.linalg.norm(curr_xy - prev_xy) <= tolerance:
            continue
        segment = next_xy - prev_xy
        if np.linalg.norm(segment) <= tolerance:
            kept.append(values[index])
            continue
        offset = curr_xy - prev_xy
        cross = segment[0] * offset[1] - segment[1] * offset[0]
        distance = abs(cross) / np.linalg.norm(segment)
        if distance > tolerance:
            kept.append(values[index])
    kept.append(values[-1])
    return np.vstack(kept)


def resample_path(path: np.ndarray, target_length: int = 64) -> np.ndarray:
    values = np.asarray(path, dtype=float)
    if values.ndim != 2:
        raise ValueError("path must have shape (path_length, path_dim)")
    if target_length < 2:
        raise ValueError("target_length must be at least 2")
    if len(values) == 1:
        return np.repeat(values, target_length, axis=0)

    segment_lengths = np.linalg.norm(np.diff(values[:, :2], axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    if cumulative[-1] <= 1e-12:
        return np.linspace(values[0], values[-1], target_length)

    target = np.linspace(0.0, cumulative[-1], target_length)
    resampled = np.zeros((target_length, values.shape[1]), dtype=float)
    for dim in range(values.shape[1]):
        resampled[:, dim] = np.interp(target, cumulative, values[:, dim])
    resampled[0] = values[0]
    resampled[-1] = values[-1]
    return resampled
