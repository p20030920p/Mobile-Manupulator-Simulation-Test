cmake_minimum_required(VERSION 3.8)
project(nmoma_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(action_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/Plan.action"
  DEPENDENCIES sensor_msgs nav_msgs trajectory_msgs action_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
