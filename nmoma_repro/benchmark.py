from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from nmoma_repro.kinematics import RobotSpec
from nmoma_repro.metrics import diversity_score, trajectory_duration
from nmoma_repro.planner import PlannerModel
from nmoma_repro.primitives import PrimitiveLibrary
from nmoma_repro.scene import generate_scene


@dataclass(frozen=True)
class BenchmarkTask:
    scene_type: str
    seed: int
    start: np.ndarray
    goal: np.ndarray


def random_state(rng: np.random.Generator, room_size: float, spec: RobotSpec) -> np.ndarray:
    state = np.zeros(spec.state_dim, dtype=float)
    state[0] = rng.uniform(-room_size * 0.35, room_size * 0.35)
    state[1] = rng.uniform(-room_size * 0.35, room_size * 0.35)
    state[2] = rng.uniform(-np.pi, np.pi)
    state[3:] = rng.uniform(-0.4, 0.4, size=spec.arm_dof)
    return state


def make_tasks(scene_types: list[str], task_count: int, seed: int, room_size: float, spec: RobotSpec) -> list[BenchmarkTask]:
    rng = np.random.default_rng(seed)
    tasks: list[BenchmarkTask] = []
    for index in range(task_count):
        scene_type = scene_types[index % len(scene_types)]
        tasks.append(
            BenchmarkTask(
                scene_type=scene_type,
                seed=seed + index,
                start=random_state(rng, room_size, spec),
                goal=random_state(rng, room_size, spec),
            )
        )
    return tasks


def _path_base_xyz(path: np.ndarray, z: float = 0.2) -> np.ndarray:
    xyz = np.zeros((path.shape[0], 3), dtype=float)
    xyz[:, :2] = path[:, :2]
    xyz[:, 2] = z
    return xyz


def evaluate_planner(
    scene_types: list[str] | None = None,
    task_count: int = 100,
    sample_count: int = 4,
    seed: int = 0,
    primitive_library: PrimitiveLibrary | None = None,
    room_size: float = 10.0,
    point_count: int = 512,
) -> dict[str, float | int | list[str]]:
    if task_count < 1:
        raise ValueError("task_count must be positive")
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    scene_types = scene_types or ["cuboids", "mixed"]
    spec = RobotSpec.default()
    planner = PlannerModel(spec=spec, primitive_library=primitive_library, seed=seed)
    tasks = make_tasks(scene_types, task_count, seed, room_size, spec)

    success_count = 0
    planning_times_ms = []
    durations = []
    diversities = []
    for task in tasks:
        scene = generate_scene(task.scene_type, seed=task.seed, room_size=room_size, point_count=point_count)
        start_time = time.perf_counter()
        candidates = planner.forward(scene.pointcloud, task.start, task.goal, sample_count=sample_count)
        planning_times_ms.append((time.perf_counter() - start_time) * 1000.0)

        safety = [float(np.min(scene.sdf(_path_base_xyz(candidate)))) for candidate in candidates]
        feasible_indices = [index for index, min_sdf in enumerate(safety) if min_sdf >= -0.02]
        if feasible_indices:
            success_count += 1
            best = candidates[feasible_indices[0]]
            durations.append(trajectory_duration(best))
        else:
            durations.append(0.0)
        diversities.append(diversity_score(candidates))

    return {
        "scene_types": scene_types,
        "task_count": task_count,
        "sample_count": sample_count,
        "S.R.": success_count / task_count * 100.0,
        "T.P./ms": float(np.mean(planning_times_ms)),
        "T.D./s": float(np.mean(durations)),
        "D.S.": float(np.mean(diversities)),
    }
