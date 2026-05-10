import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nmoma_repro.data import load_sample, save_sample
from nmoma_repro.kinematics import RobotSpec, boundary_keypoint_sequence, state_to_path_point
from nmoma_repro.metrics import diversity_score, path_length
from nmoma_repro.planner import PlannerModel
from nmoma_repro.primitives import PrimitiveLibrary, build_primitive_library
from nmoma_repro.scene import generate_scene


class CoreMvpTest(unittest.TestCase):
    def test_keypoint_sequence_maps_start_and_goal_into_ordered_3d_features(self):
        spec = RobotSpec.default()
        start = np.array([0.0, 0.0, 0.0, 0.0, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3])
        goal = np.array([1.0, 1.0, np.pi / 2, 0.2, 0.1, 0.0, -0.1, 0.2, -0.2, 0.1])

        keypoints = boundary_keypoint_sequence(start, goal, spec)

        self.assertEqual(keypoints.shape, (2 * (spec.arm_dof + 2), 4))
        np.testing.assert_allclose(keypoints[0, :3], [0.0, 0.0, 0.0], atol=1e-7)
        np.testing.assert_allclose(keypoints[spec.arm_dof + 2, :3], [1.0, 1.0, 0.0], atol=1e-7)
        self.assertAlmostEqual(keypoints[0, 3], 1.0)
        self.assertAlmostEqual(keypoints[spec.arm_dof + 2, 3], 0.0, places=7)

    def test_scene_generation_returns_pointcloud_and_signed_distance_field(self):
        scene = generate_scene("cuboids", seed=7, room_size=4.0, point_count=256)

        self.assertEqual(scene.pointcloud.shape, (256, 3))
        self.assertGreater(len(scene.obstacles), 0)
        outside = scene.sdf(np.array([[1.8, 1.8, 1.5]]))[0]
        inside = scene.sdf(np.array([scene.obstacles[0].center]))[0]
        self.assertGreater(outside, -0.25)
        self.assertLess(inside, 0.05)

    def test_primitives_cluster_paths_and_select_nearest_candidate(self):
        start = np.zeros(10)
        goal_a = np.ones(10)
        goal_b = np.array([2.0] * 10)
        path_a = np.linspace(state_to_path_point(start), state_to_path_point(goal_a), 64)
        path_b = np.linspace(state_to_path_point(start), state_to_path_point(goal_b), 64)

        library = build_primitive_library(np.stack([path_a, path_a + 0.01, path_b]), primitive_count=2, seed=3)
        idx = library.nearest(path_a)

        self.assertIsInstance(library, PrimitiveLibrary)
        self.assertEqual(library.primitives.shape, (2, 64, 11))
        self.assertIn(idx, [0, 1])

    def test_planner_model_forward_returns_start_goal_conditioned_candidate_paths(self):
        spec = RobotSpec.default()
        scene = generate_scene("mixed", seed=11, room_size=5.0, point_count=128)
        start = np.array([-1.0, -1.0, 0.0, 0.0, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3])
        goal = np.array([1.0, 1.0, np.pi / 3, 0.2, 0.1, 0.0, -0.1, 0.2, -0.2, 0.1])
        start_point = state_to_path_point(start)
        goal_point = state_to_path_point(goal)
        primitive = np.linspace(start_point, goal_point, 64)

        planner = PlannerModel(spec=spec, primitive_library=PrimitiveLibrary(np.expand_dims(primitive, axis=0)), seed=5)
        candidates = planner.forward(scene.pointcloud, start, goal, sample_count=4)

        self.assertEqual(candidates.shape, (4, 64, 11))
        np.testing.assert_allclose(candidates[:, 0, :], np.tile(start_point, (4, 1)), atol=1e-7)
        np.testing.assert_allclose(candidates[:, -1, :], np.tile(goal_point, (4, 1)), atol=1e-7)
        self.assertGreaterEqual(diversity_score(candidates), 0.0)
        self.assertGreater(path_length(candidates[0]), 0.0)

    def test_sample_round_trip_matches_expected_file_contract(self):
        scene = generate_scene("cuboids", seed=13, room_size=4.0, point_count=32)
        start = np.zeros(10)
        goal = np.ones(10)
        path = np.linspace(state_to_path_point(start), state_to_path_point(goal), 64)
        metadata = {"scene_type": "cuboids", "planner": "mvp"}

        with tempfile.TemporaryDirectory() as tmp:
            sample_dir = Path(tmp) / "sample_000001"
            save_sample(sample_dir, scene.pointcloud, scene.sdf_grid(grid_size=8), start, goal, path, metadata)
            loaded = load_sample(sample_dir)

            self.assertTrue((sample_dir / "pointcloud.npy").exists())
            self.assertTrue((sample_dir / "sdf.npy").exists())
            self.assertTrue((sample_dir / "start.npy").exists())
            self.assertTrue((sample_dir / "goal.npy").exists())
            self.assertTrue((sample_dir / "path_64.npy").exists())
            self.assertEqual(json.loads((sample_dir / "trajectory_meta.json").read_text())["planner"], "mvp")
            np.testing.assert_allclose(loaded["path"], path)


if __name__ == "__main__":
    unittest.main()
