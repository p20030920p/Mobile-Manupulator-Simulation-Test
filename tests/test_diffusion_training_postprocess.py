import unittest

import numpy as np

from nmoma_repro.diffusion import DiffusionSchedule, diffuse_from_primitive
from nmoma_repro.kinematics import state_to_path_point
from nmoma_repro.losses import reconstruction_loss, smoothness_loss, uniform_loss
from nmoma_repro.optimizer import prune_path, resample_path
from nmoma_repro.scene import generate_scene


class DiffusionTrainingPostprocessTest(unittest.TestCase):
    def test_truncated_diffusion_adds_step_scaled_noise_to_primitive(self):
        primitive = np.ones((64, 11))
        schedule = DiffusionSchedule.linear(total_steps=10, beta_start=0.01, beta_end=0.02)
        noise = np.zeros_like(primitive)

        diffused = diffuse_from_primitive(primitive, schedule=schedule, truncated_step=4, noise=noise)

        expected_scale = np.sqrt(schedule.alpha_bars[3])
        np.testing.assert_allclose(diffused, primitive * expected_scale)

    def test_training_losses_reward_reconstruction_smoothness_and_uniform_arc_length(self):
        start = np.zeros(10)
        goal = np.ones(10)
        straight = np.linspace(state_to_path_point(start), state_to_path_point(goal), 64)
        perturbed = straight.copy()
        perturbed[32, 0] += 0.25

        self.assertAlmostEqual(reconstruction_loss(straight, straight), 0.0)
        self.assertGreater(reconstruction_loss(perturbed, straight), 0.0)
        self.assertGreater(smoothness_loss(perturbed), smoothness_loss(straight))
        self.assertLess(uniform_loss(straight), 1e-10)

    def test_prune_and_resample_preserve_endpoints_and_path_length_contract(self):
        start = np.zeros(10)
        goal = np.ones(10)
        dense = np.linspace(state_to_path_point(start), state_to_path_point(goal), 64)
        dense[20:40, :2] = dense[20, :2]

        pruned = prune_path(dense, tolerance=1e-9)
        resampled = resample_path(pruned, target_length=64)

        self.assertLess(len(pruned), len(dense))
        self.assertEqual(resampled.shape, dense.shape)
        np.testing.assert_allclose(resampled[0], dense[0])
        np.testing.assert_allclose(resampled[-1], dense[-1])

    def test_scene_sdf_can_score_candidate_path_safety(self):
        scene = generate_scene("cuboids", seed=21, room_size=4.0, point_count=64)
        path = np.zeros((64, 11))
        path[:, :3] = np.array(scene.obstacles[0].center)

        self.assertLess(scene.sdf(path[:, :3]).min(), 0.0)


if __name__ == "__main__":
    unittest.main()
