from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nmoma_repro.benchmark import evaluate_planner
from nmoma_repro.primitives import PrimitiveLibrary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small NMOMA MVP benchmark.")
    parser.add_argument("--scenes", default="cuboids,mixed", help="Comma-separated scene list.")
    parser.add_argument("--task-count", type=int, default=20)
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--primitive-library", default=None, help="Optional primitive library .npz path.")
    args = parser.parse_args()

    library = PrimitiveLibrary.load(args.primitive_library) if args.primitive_library else None
    report = evaluate_planner(
        scene_types=[scene.strip() for scene in args.scenes.split(",") if scene.strip()],
        task_count=args.task_count,
        sample_count=args.sample_count,
        seed=args.seed,
        primitive_library=library,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
