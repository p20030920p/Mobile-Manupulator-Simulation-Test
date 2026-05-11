import tempfile
import unittest
from pathlib import Path

import numpy as np

from nmoma_repro.benchmark import evaluate_planner
from nmoma_repro.data import save_sample
from nmoma_repro.kinematics import RobotSpec, state_to_path_point
from nmoma_repro.primitives import PrimitiveLibrary
from nmoma_repro.scene import generate_scene
from nmoma_repro.training import fit_primitive_library_from_dataset


ROOT = Path(__file__).resolve().parents[1]


class TrainingBenchmarkScaffoldTest(unittest.TestCase):
    def test_primitive_library_round_trips_to_npz(self):
        library = PrimitiveLibrary(np.ones((2, 64, 11)))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "primitives.npz"
            library.save(path)
            loaded = PrimitiveLibrary.load(path)

        np.testing.assert_allclose(loaded.primitives, library.primitives)

    def test_fit_primitives_from_dataset_uses_path_64_files(self):
        spec = RobotSpec.default()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in range(3):
                scene = generate_scene("cuboids", seed=index, point_count=32)
                start = np.zeros(spec.state_dim)
                goal = np.ones(spec.state_dim) * (index + 1)
                path = np.linspace(state_to_path_point(start, spec), state_to_path_point(goal, spec), 64)
                save_sample(root / f"sample_{index:06d}", scene.pointcloud, scene.sdf_grid(4), start, goal, path, {})

            library = fit_primitive_library_from_dataset(root, primitive_count=2, seed=0)

        self.assertEqual(library.primitives.shape, (2, 64, 11))

    def test_benchmark_returns_paper_metric_contract(self):
        report = evaluate_planner(scene_types=["cuboids"], task_count=3, sample_count=3, seed=2)

        self.assertEqual(report["task_count"], 3)
        self.assertIn("S.R.", report)
        self.assertIn("T.P./ms", report)
        self.assertIn("T.D./s", report)
        self.assertIn("D.S.", report)
        self.assertGreaterEqual(report["S.R."], 0.0)
        self.assertLessEqual(report["S.R."], 100.0)
        self.assertGreaterEqual(report["T.P./ms"], 0.0)
        self.assertGreaterEqual(report["T.D./s"], 0.0)
        self.assertGreaterEqual(report["D.S."], 0.0)

    def test_training_and_benchmark_scripts_exist(self):
        self.assertTrue((ROOT / "scripts" / "fit_primitives.py").exists())
        self.assertTrue((ROOT / "scripts" / "run_benchmark.py").exists())


if __name__ == "__main__":
    unittest.main()
