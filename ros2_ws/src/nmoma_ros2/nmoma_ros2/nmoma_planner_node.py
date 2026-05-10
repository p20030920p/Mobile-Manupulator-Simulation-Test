from __future__ import annotations

import numpy as np

from nmoma_repro.kinematics import RobotSpec
from nmoma_repro.planner import PlannerModel


def _lazy_ros_imports():
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Path
    from nmoma_msgs.action import Plan
    from rclpy.action import ActionServer
    from rclpy.node import Node
    from sensor_msgs_py import point_cloud2
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    return {
        "rclpy": rclpy,
        "PoseStamped": PoseStamped,
        "Path": Path,
        "Plan": Plan,
        "ActionServer": ActionServer,
        "Node": Node,
        "point_cloud2": point_cloud2,
        "JointTrajectory": JointTrajectory,
        "JointTrajectoryPoint": JointTrajectoryPoint,
    }


class NmomaPlannerNode:
    """Factory wrapper so this file remains importable without ROS installed."""

    @staticmethod
    def create():
        ros = _lazy_ros_imports()
        Node = ros["Node"]
        ActionServer = ros["ActionServer"]
        Plan = ros["Plan"]

        class _Node(Node):
            def __init__(self):
                super().__init__("nmoma_planner_node")
                self.spec = RobotSpec.default()
                self.planner = PlannerModel(spec=self.spec, seed=0)
                self.server = ActionServer(self, Plan, "/nmoma/plan", self.execute_callback)
                self.get_logger().info("NMOMA planner action server ready at /nmoma/plan")

            def execute_callback(self, goal_handle):
                request = goal_handle.request
                pointcloud = np.array(
                    list(ros["point_cloud2"].read_points(request.scene, field_names=("x", "y", "z"), skip_nans=True)),
                    dtype=float,
                )
                if pointcloud.size == 0:
                    pointcloud = np.zeros((1, 3), dtype=float)
                candidates = self.planner.forward(
                    pointcloud,
                    np.asarray(request.start_state, dtype=float),
                    np.asarray(request.goal_state, dtype=float),
                    max(1, int(request.sample_count)),
                )
                best = candidates[0]
                result = Plan.Result()
                result.base_path = self._to_base_path(best)
                result.arm_trajectory = self._to_arm_trajectory(best)
                result.success = True
                result.message = "Generated MVP NMOMA candidate trajectory."
                goal_handle.succeed()
                return result

            def _to_base_path(self, candidate):
                PathMsg = ros["Path"]
                PoseStamped = ros["PoseStamped"]
                path_msg = PathMsg()
                path_msg.header.frame_id = "map"
                for row in candidate:
                    pose = PoseStamped()
                    pose.header.frame_id = "map"
                    pose.pose.position.x = float(row[0])
                    pose.pose.position.y = float(row[1])
                    pose.pose.orientation.z = float(row[3])
                    pose.pose.orientation.w = float(row[2])
                    path_msg.poses.append(pose)
                return path_msg

            def _to_arm_trajectory(self, candidate):
                JointTrajectory = ros["JointTrajectory"]
                JointTrajectoryPoint = ros["JointTrajectoryPoint"]
                trajectory = JointTrajectory()
                trajectory.joint_names = [f"arm_joint_{i}" for i in range(1, self.spec.arm_dof + 1)]
                for index, row in enumerate(candidate):
                    point = JointTrajectoryPoint()
                    point.positions = [float(v) for v in row[4:]]
                    point.time_from_start.sec = int(index * 0.1)
                    point.time_from_start.nanosec = int((index * 0.1 - int(index * 0.1)) * 1e9)
                    trajectory.points.append(point)
                return trajectory

        return _Node()


def main(args=None):
    ros = _lazy_ros_imports()
    rclpy = ros["rclpy"]
    rclpy.init(args=args)
    node = NmomaPlannerNode.create()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
