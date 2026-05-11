from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nmoma_repro.training import fit_primitive_library_from_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit NMOMA motion primitives from path_64 dataset samples.")
    parser.add_argument("--dataset", required=True, help="Dataset root containing sample_*/path_64.npy files.")
    parser.add_argument("--out", default="artifacts/primitives_mvp.npz", help="Output primitive library .npz path.")
    parser.add_argument("--primitive-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    library = fit_primitive_library_from_dataset(
        args.dataset,
        primitive_count=args.primitive_count,
        seed=args.seed,
        output_path=args.out,
    )
    print(f"Wrote primitive library {args.out} with shape {library.primitives.shape}")


if __name__ == "__main__":
    main()
