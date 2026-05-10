from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nmoma_repro.kinematics import RobotSpec, state_to_path_point
from nmoma_repro.metrics import diversity_score, path_length, trajectory_duration
from nmoma_repro.planner import PlannerModel
from nmoma_repro.primitives import PrimitiveLibrary
from nmoma_repro.scene import generate_scene


def main() -> None:
    spec = RobotSpec.default()
    scene = generate_scene("cuboids", seed=42, point_count=512)
    start = np.array([-2.0, -2.0, 0.0, 0.0, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3])
    goal = np.array([2.0, 2.0, np.pi / 2.0, 0.2, 0.1, 0.0, -0.1, 0.2, -0.2, 0.1])
    seed_path = np.linspace(state_to_path_point(start, spec), state_to_path_point(goal, spec), 64)
    planner = PlannerModel(spec=spec, primitive_library=PrimitiveLibrary(np.expand_dims(seed_path, axis=0)), seed=9)
    candidates = planner.forward(scene.pointcloud, start, goal, sample_count=4)
    report = {
        "candidate_shape": list(candidates.shape),
        "diversity_score": diversity_score(candidates),
        "first_path_length": path_length(candidates[0]),
        "first_trajectory_duration_at_0_5mps": trajectory_duration(candidates[0]),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
