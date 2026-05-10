from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nmoma_repro.kinematics import RobotSpec, state_to_path_point
from nmoma_repro.primitives import PrimitiveLibrary


def linear_path(start: np.ndarray, goal: np.ndarray, path_length: int = 64, spec: RobotSpec | None = None) -> np.ndarray:
    start_point = state_to_path_point(start, spec)
    goal_point = state_to_path_point(goal, spec)
    return np.linspace(start_point, goal_point, path_length)


@dataclass
class PlannerModel:
    """Numpy MVP with the same public shape as the planned PTDM model."""

    spec: RobotSpec
    primitive_library: PrimitiveLibrary | None = None
    seed: int = 0
    noise_scale: float = 0.025

    def forward(
        self,
        pointcloud: np.ndarray,
        start_state: np.ndarray,
        goal_state: np.ndarray,
        sample_count: int = 4,
    ) -> np.ndarray:
        if sample_count < 1:
            raise ValueError("sample_count must be positive")
        pointcloud = np.asarray(pointcloud, dtype=float)
        if pointcloud.ndim != 2 or pointcloud.shape[1] != 3:
            raise ValueError("pointcloud must have shape (N, 3)")

        base_path = linear_path(start_state, goal_state, 64, self.spec)
        rng = np.random.default_rng(self.seed)
        candidates = []
        primitive_shape = np.zeros_like(base_path)

        if self.primitive_library is not None:
            primitive_index = self.primitive_library.nearest(base_path)
            primitive = self.primitive_library.primitives[primitive_index]
            primitive_linear = np.linspace(primitive[0], primitive[-1], primitive.shape[0])
            primitive_shape = primitive - primitive_linear

        for sample_index in range(sample_count):
            phase = np.linspace(0.0, np.pi, base_path.shape[0])
            envelope = np.sin(phase)[:, None]
            lateral = np.zeros_like(base_path)
            lateral[:, 0] = np.sin(phase * (sample_index + 1)) * self.noise_scale * (sample_index + 1)
            lateral[:, 1] = np.cos(phase * (sample_index + 1)) * self.noise_scale * (sample_index + 1)
            joint_noise = rng.normal(0.0, self.noise_scale * 0.2, size=base_path[:, 4:].shape)
            candidate = base_path + primitive_shape * envelope + lateral * envelope
            candidate[:, 4:] += joint_noise * envelope
            candidate[0] = base_path[0]
            candidate[-1] = base_path[-1]
            candidates.append(candidate)

        return np.stack(candidates, axis=0)
