import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ScaffoldContractsTest(unittest.TestCase):
    def test_mujoco_model_is_parseable_and_has_seven_arm_joints(self):
        model_path = ROOT / "mujoco" / "ddmoma_7dof.xml"
        tree = ET.parse(model_path)
        joint_names = [joint.attrib.get("name", "") for joint in tree.findall(".//joint")]

        self.assertIn("base_free", joint_names)
        self.assertEqual(len([name for name in joint_names if name.startswith("arm_joint_")]), 7)

    def test_ros2_plan_action_exposes_fixed_reproduction_interface(self):
        action_path = ROOT / "ros2_ws" / "src" / "nmoma_msgs" / "action" / "Plan.action"
        text = action_path.read_text(encoding="utf-8")

        self.assertIn("float64[] start_state", text)
        self.assertIn("float64[] goal_state", text)
        self.assertIn("sensor_msgs/PointCloud2 scene", text)
        self.assertIn("nav_msgs/Path base_path", text)
        self.assertIn("trajectory_msgs/JointTrajectory arm_trajectory", text)

    def test_demo_scripts_are_present_for_dataset_and_core_planner(self):
        self.assertTrue((ROOT / "scripts" / "generate_mvp_dataset.py").exists())
        self.assertTrue((ROOT / "scripts" / "run_core_planner_demo.py").exists())
        self.assertTrue((ROOT / "scripts" / "run_mujoco_demo.py").exists())


if __name__ == "__main__":
    unittest.main()
