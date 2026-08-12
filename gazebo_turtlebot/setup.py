from setuptools import find_packages, setup


package_name = "gazebo_turtlebot"


setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        (
            "share/" + package_name,
            ["package.xml"],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="anonymous",
    maintainer_email="anonymous@example.com",
    description="TurtleBot3 Gazebo execution layer for grid navigation experiments.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "waypoint_driver = gazebo_turtlebot.waypoint_driver:main",
            "grid_action = gazebo_turtlebot.grid_action_controller:main",
        ],
    },
)
