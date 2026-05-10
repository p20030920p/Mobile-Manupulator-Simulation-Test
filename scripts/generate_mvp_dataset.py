from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nmoma_repro.data import save_sample
from nmoma_repro.kinematics import RobotSpec, state_to_path_point
from nmoma_repro.scene import generate_scene


def make_state(rng: np.random.Generator, room_size: float, spec: RobotSpec) -> np.ndarray:
    state = np.zeros(spec.state_dim, dtype=float)
    state[0] = rng.uniform(-room_size * 0.35, room_size * 0.35)
    state[1] = rng.uniform(-room_size * 0.35, room_size * 0.35)
    state[2] = rng.uniform(-np.pi, np.pi)
    state[3:] = rng.uniform(-0.4, 0.4, size=spec.arm_dof)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a small NMOMA MVP dataset.")
    parser.add_argument("--out", default="datasets/mvp_cuboids", help="Output dataset directory.")
    parser.add_argument("--count", type=int, default=16, help="Number of samples to generate.")
    parser.add_argument("--scene", choices=["cuboids", "mixed"], default="cuboids")
    parser.add_argument("--room-size", type=float, default=10.0)
    parser.add_argument("--point-count", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    root = Path(args.out)
    spec = RobotSpec.default()
    for index in range(args.count):
        scene = generate_scene(args.scene, seed=args.seed + index, room_size=args.room_size, point_count=args.point_count)
        start = make_state(rng, args.room_size, spec)
        goal = make_state(rng, args.room_size, spec)
        path = np.linspace(state_to_path_point(start, spec), state_to_path_point(goal, spec), 64)
        metadata = {
            "scene_type": args.scene,
            "planner": "linear_mvp_seed",
            "sample_index": index,
            "paper_path_contract": "[x, y, cos(theta), sin(theta), q1..q7] x 64",
        }
        save_sample(root / f"sample_{index:06d}", scene.pointcloud, scene.sdf_grid(grid_size=16), start, goal, path, metadata)
    print(f"Wrote {args.count} samples to {root}")


if __name__ == "__main__":
    main()
