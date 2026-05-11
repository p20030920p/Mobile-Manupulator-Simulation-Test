import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.run_mujoco_demo import demo_joint_targets, write_png_frame, write_ppm_frame


ROOT = Path(__file__).resolve().parents[1]


class MujocoVisualDemoTest(unittest.TestCase):
    def test_demo_joint_targets_move_planar_base_and_all_arm_joints(self):
        first = demo_joint_targets(0.0)
        later = demo_joint_targets(0.5)

        self.assertIn("base_x", first)
        self.assertIn("base_y", first)
        self.assertIn("base_yaw", first)
        self.assertEqual(len([name for name in first if name.startswith("arm_joint_")]), 7)
        self.assertNotEqual(first["base_x"], later["base_x"])
        self.assertNotEqual(first["arm_joint_3"], later["arm_joint_3"])

    def test_write_ppm_frame_creates_viewable_image_without_extra_dependencies(self):
        image = np.zeros((2, 3, 3), dtype=np.uint8)
        image[0, 0] = [255, 0, 0]

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "frame_0000.ppm"
            write_ppm_frame(output, image)
            data = output.read_bytes()

        self.assertTrue(data.startswith(b"P6\n3 2\n255\n"))
        self.assertIn(bytes([255, 0, 0]), data)

    def test_write_png_frame_creates_viewable_image_without_extra_dependencies(self):
        image = np.zeros((2, 3, 3), dtype=np.uint8)
        image[0, 0] = [0, 255, 0]

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "frame_0000.png"
            write_png_frame(output, image)
            data = output.read_bytes()

        self.assertTrue(data.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertIn(b"IHDR", data)
        self.assertIn(b"IDAT", data)

    def test_readme_documents_visual_demo_commands(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("--viewer", readme)
        self.assertIn("--render-dir artifacts/mujoco_demo_frames", readme)
        self.assertIn("--frame-format png", readme)

    def test_script_exposes_viewer_and_render_dir_flags(self):
        text = (ROOT / "scripts" / "run_mujoco_demo.py").read_text(encoding="utf-8")

        self.assertIn("--viewer", text)
        self.assertIn("--render-dir", text)
        self.assertIn("--frame-format", text)
        self.assertIn("mujoco.viewer", text)
        self.assertIn("Renderer", text)


if __name__ == "__main__":
    unittest.main()
