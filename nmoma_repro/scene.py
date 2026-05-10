from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Obstacle:
    kind: str
    center: np.ndarray
    size: np.ndarray

    def sdf(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        if self.kind == "sphere":
            return np.linalg.norm(points - self.center, axis=1) - float(self.size[0])
        if self.kind == "cylinder":
            radial = np.linalg.norm(points[:, :2] - self.center[:2], axis=1) - float(self.size[0])
            vertical = np.abs(points[:, 2] - self.center[2]) - float(self.size[1]) / 2.0
            outside = np.linalg.norm(np.maximum(np.stack([radial, vertical], axis=1), 0.0), axis=1)
            inside = np.minimum(np.maximum(radial, vertical), 0.0)
            return outside + inside
        if self.kind == "box":
            q = np.abs(points - self.center) - self.size / 2.0
            outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
            inside = np.minimum(np.max(q, axis=1), 0.0)
            return outside + inside
        raise ValueError(f"unsupported obstacle kind: {self.kind}")


@dataclass
class Scene:
    scene_type: str
    room_size: float
    obstacles: list[Obstacle]
    pointcloud: np.ndarray

    def sdf(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, 3)
        if not self.obstacles:
            return np.full(points.shape[0], self.room_size, dtype=float)
        distances = np.vstack([obs.sdf(points) for obs in self.obstacles])
        return np.min(distances, axis=0)

    def sdf_grid(self, grid_size: int = 32, z_max: float = 2.0) -> np.ndarray:
        xs = np.linspace(-self.room_size / 2.0, self.room_size / 2.0, grid_size)
        ys = np.linspace(-self.room_size / 2.0, self.room_size / 2.0, grid_size)
        zs = np.linspace(0.0, z_max, grid_size)
        grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
        return self.sdf(grid).reshape(grid_size, grid_size, grid_size)


def _sample_box_surface(rng: np.random.Generator, obstacle: Obstacle, count: int) -> np.ndarray:
    local = rng.uniform(-0.5, 0.5, size=(count, 3)) * obstacle.size
    face_axis = rng.integers(0, 3, size=count)
    face_sign = rng.choice([-1.0, 1.0], size=count)
    local[np.arange(count), face_axis] = face_sign * obstacle.size[face_axis] / 2.0
    return obstacle.center + local


def _sample_sphere_surface(rng: np.random.Generator, obstacle: Obstacle, count: int) -> np.ndarray:
    direction = rng.normal(size=(count, 3))
    direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
    return obstacle.center + direction * float(obstacle.size[0])


def _sample_cylinder_surface(rng: np.random.Generator, obstacle: Obstacle, count: int) -> np.ndarray:
    angle = rng.uniform(0.0, 2.0 * np.pi, size=count)
    z = rng.uniform(-obstacle.size[1] / 2.0, obstacle.size[1] / 2.0, size=count)
    return np.column_stack(
        [
            obstacle.center[0] + np.cos(angle) * obstacle.size[0],
            obstacle.center[1] + np.sin(angle) * obstacle.size[0],
            obstacle.center[2] + z,
        ]
    )


def _obstacle_pointcloud(rng: np.random.Generator, obstacles: Iterable[Obstacle], point_count: int) -> np.ndarray:
    obstacles = list(obstacles)
    per_obstacle = max(1, int(np.ceil(point_count / len(obstacles))))
    chunks = []
    for obstacle in obstacles:
        if obstacle.kind == "box":
            chunks.append(_sample_box_surface(rng, obstacle, per_obstacle))
        elif obstacle.kind == "sphere":
            chunks.append(_sample_sphere_surface(rng, obstacle, per_obstacle))
        elif obstacle.kind == "cylinder":
            chunks.append(_sample_cylinder_surface(rng, obstacle, per_obstacle))
    pointcloud = np.vstack(chunks)
    rng.shuffle(pointcloud)
    return pointcloud[:point_count]


def generate_scene(scene_type: str, seed: int = 0, room_size: float = 10.0, point_count: int = 2048) -> Scene:
    rng = np.random.default_rng(seed)
    scene_key = scene_type.lower()
    if scene_key not in {"cuboids", "mixed"}:
        raise ValueError("scene_type must be 'cuboids' or 'mixed' for the MVP")

    obstacles: list[Obstacle] = []
    span = room_size * 0.34
    if scene_key == "cuboids":
        for _ in range(8):
            center = np.array([rng.uniform(-span, span), rng.uniform(-span, span), rng.uniform(0.35, 1.45)])
            size = np.array([rng.uniform(0.35, 0.75), rng.uniform(0.35, 0.75), rng.uniform(0.35, 1.0)])
            obstacles.append(Obstacle("box", center, size))
    else:
        for _ in range(4):
            center = np.array([rng.uniform(-span, span), rng.uniform(-span, span), rng.uniform(0.35, 1.35)])
            size = np.array([rng.uniform(0.35, 0.75), rng.uniform(0.35, 0.75), rng.uniform(0.35, 0.95)])
            obstacles.append(Obstacle("box", center, size))
        for _ in range(3):
            center = np.array([rng.uniform(-span, span), rng.uniform(-span, span), rng.uniform(0.45, 1.55)])
            obstacles.append(Obstacle("sphere", center, np.array([rng.uniform(0.22, 0.45)])))
        for _ in range(3):
            center = np.array([rng.uniform(-span, span), rng.uniform(-span, span), rng.uniform(0.55, 1.15)])
            obstacles.append(Obstacle("cylinder", center, np.array([rng.uniform(0.18, 0.38), rng.uniform(0.5, 1.4)])))

    return Scene(scene_key, room_size, obstacles, _obstacle_pointcloud(rng, obstacles, point_count))
