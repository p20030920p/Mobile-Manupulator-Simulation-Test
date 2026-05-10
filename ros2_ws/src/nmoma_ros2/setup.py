from setuptools import setup

package_name = "nmoma_ros2"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="RAS Lab",
    maintainer_email="user@example.com",
    description="ROS 2 action server wrapper for the NMOMA reproduction planner.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "nmoma_planner_node = nmoma_ros2.nmoma_planner_node:main",
        ],
    },
)
