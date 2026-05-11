import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ScaffoldContractsTest(unittest.TestCase):
    def test_mujoco_model_is_parseable_and_has_planar_base_and_seven_arm_joints(self):
        model_path = ROOT / "mujoco_models" / "ddmoma_7dof.xml"
        tree = ET.parse(model_path)
        joint_names = [joint.attrib.get("name", "") for joint in tree.findall(".//joint")]

        self.assertNotIn("base_free", joint_names)
        self.assertIn("base_x", joint_names)
        self.assertIn("base_y", joint_names)
        self.assertIn("base_yaw", joint_names)
        self.assertEqual(len([name for name in joint_names if name.startswith("arm_joint_")]), 7)

    def test_mujoco_wheel_actuators_do_not_lock_planar_base_with_high_gain(self):
        model_path = ROOT / "mujoco_models" / "ddmoma_7dof.xml"
        tree = ET.parse(model_path)
        wheel_actuators = [
            actuator
            for actuator in tree.findall(".//actuator/velocity")
            if actuator.attrib.get("joint", "").endswith("wheel_joint")
        ]

        self.assertEqual(len(wheel_actuators), 2)
        self.assertTrue(all(float(actuator.attrib["kv"]) <= 5.0 for actuator in wheel_actuators))

    def test_mujoco_demo_checks_for_numerical_instability(self):
        demo_path = ROOT / "scripts" / "run_mujoco_demo.py"
        text = demo_path.read_text(encoding="utf-8")

        self.assertIn("assert_finite_state", text)
        self.assertIn("qacc", text)
        self.assertIn("Simulation became unstable", text)

    def test_mujoco_demo_loads_xml_through_python_for_unicode_paths(self):
        demo_path = ROOT / "scripts" / "run_mujoco_demo.py"
        text = demo_path.read_text(encoding="utf-8")

        self.assertIn("read_text", text)
        self.assertIn("from_xml_string", text)

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
