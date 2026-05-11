from __future__ import annotations

from pathlib import Path

import numpy as np

from nmoma_repro.primitives import PrimitiveLibrary, build_primitive_library


def load_dataset_paths(dataset_dir: str | Path) -> np.ndarray:
    root = Path(dataset_dir)
    path_files = sorted(root.glob("sample_*/path_64.npy"))
    if not path_files:
        raise FileNotFoundError(f"No sample_*/path_64.npy files found under {root}")
    paths = [np.load(path_file) for path_file in path_files]
    return np.stack(paths, axis=0)


def fit_primitive_library_from_dataset(
    dataset_dir: str | Path,
    primitive_count: int = 32,
    seed: int = 0,
    output_path: str | Path | None = None,
) -> PrimitiveLibrary:
    paths = load_dataset_paths(dataset_dir)
    library = build_primitive_library(paths, primitive_count=primitive_count, seed=seed)
    if output_path is not None:
        library.save(output_path)
    return library
