from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def save_sample(
    sample_dir: str | Path,
    pointcloud: np.ndarray,
    sdf: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    path: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    root = Path(sample_dir)
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "pointcloud.npy", np.asarray(pointcloud, dtype=float))
    np.save(root / "sdf.npy", np.asarray(sdf, dtype=float))
    np.save(root / "start.npy", np.asarray(start, dtype=float))
    np.save(root / "goal.npy", np.asarray(goal, dtype=float))
    np.save(root / "path_64.npy", np.asarray(path, dtype=float))
    (root / "trajectory_meta.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def load_sample(sample_dir: str | Path) -> dict[str, Any]:
    root = Path(sample_dir)
    return {
        "pointcloud": np.load(root / "pointcloud.npy"),
        "sdf": np.load(root / "sdf.npy"),
        "start": np.load(root / "start.npy"),
        "goal": np.load(root / "goal.npy"),
        "path": np.load(root / "path_64.npy"),
        "metadata": json.loads((root / "trajectory_meta.json").read_text(encoding="utf-8")),
    }
