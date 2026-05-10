from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RobotSpec:
    """Simple differential-drive mobile manipulator model used by the MVP."""

    arm_dof: int
    joint_limits: np.ndarray
    link_vectors: np.ndarray

    @staticmethod
    def default() -> "RobotSpec":
        arm_dof = 7
        joint_limits = np.array([[-np.pi, np.pi]] * arm_dof, dtype=float)
        link_vectors = np.array(
            [
                [0.30, 0.00, 0.20],
                [0.00, 0.00, 0.18],
                [0.00, 0.00, 0.22],
                [0.18, 0.00, 0.00],
                [0.00, 0.00, 0.20],
                [0.16, 0.00, 0.00],
                [0.00, 0.00, 0.16],
                [0.12, 0.00, 0.00],
            ],
            dtype=float,
        )
        return RobotSpec(arm_dof=arm_dof, joint_limits=joint_limits, link_vectors=link_vectors)

    @property
    def state_dim(self) -> int:
        return 3 + self.arm_dof

    @property
    def path_dim(self) -> int:
        return 4 + self.arm_dof


def _as_state(state: np.ndarray, spec: RobotSpec | None = None) -> np.ndarray:
    values = np.asarray(state, dtype=float)
    expected = spec.state_dim if spec else 10
    if values.shape != (expected,):
        raise ValueError(f"expected state shape ({expected},), got {values.shape}")
    return values


def _rot_z(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def normalize_joints(q: np.ndarray, spec: RobotSpec) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    lo = spec.joint_limits[:, 0]
    hi = spec.joint_limits[:, 1]
    return (2.0 * q - hi - lo) / (hi - lo)


def state_to_path_point(state: np.ndarray, spec: RobotSpec | None = None) -> np.ndarray:
    spec = spec or RobotSpec.default()
    values = _as_state(state, spec)
    x, y, theta = values[:3]
    joints = values[3:]
    return np.concatenate([[x, y, np.cos(theta), np.sin(theta)], joints])


def path_point_to_state(point: np.ndarray, spec: RobotSpec | None = None) -> np.ndarray:
    spec = spec or RobotSpec.default()
    values = np.asarray(point, dtype=float)
    if values.shape != (spec.path_dim,):
        raise ValueError(f"expected path point shape ({spec.path_dim},), got {values.shape}")
    theta = np.arctan2(values[3], values[2])
    return np.concatenate([[values[0], values[1], theta], values[4:]])


def state_to_keypoints(state: np.ndarray, spec: RobotSpec | None = None) -> np.ndarray:
    spec = spec or RobotSpec.default()
    values = _as_state(state, spec)
    x, y, theta = values[:3]
    joints = values[3:]
    joint_features = normalize_joints(joints, spec)

    keypoints = []
    base = np.array([x, y, 0.0], dtype=float)
    keypoints.append(np.array([base[0], base[1], base[2], np.cos(theta)], dtype=float))

    position = base + _rot_z(theta) @ spec.link_vectors[0]
    keypoints.append(np.array([position[0], position[1], position[2], np.sin(theta)], dtype=float))

    accumulated_angle = theta
    for joint_index in range(spec.arm_dof):
        accumulated_angle += joints[joint_index]
        link = spec.link_vectors[joint_index + 1]
        position = position + _rot_z(accumulated_angle) @ link
        keypoints.append(
            np.array(
                [
                    position[0],
                    position[1],
                    position[2],
                    joint_features[joint_index],
                ],
                dtype=float,
            )
        )

    return np.vstack(keypoints)


def boundary_keypoint_sequence(start: np.ndarray, goal: np.ndarray, spec: RobotSpec | None = None) -> np.ndarray:
    spec = spec or RobotSpec.default()
    return np.vstack([state_to_keypoints(start, spec), state_to_keypoints(goal, spec)])
