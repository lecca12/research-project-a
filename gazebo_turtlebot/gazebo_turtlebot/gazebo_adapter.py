import math

import rclpy

from gazebo_turtlebot.grid_action_controller import GridActionController


class GazeboAdapter:
    """
    Persistent interface between the navigation experiment and TurtleBot3/Gazebo.

    One GazeboAdapter instance owns one GridActionController and can execute
    multiple cardinal actions without restarting ROS between actions.
    """

    def __init__(self):
        if not rclpy.ok():
            rclpy.init()

        self.controller = GridActionController()

        self.controller.get_logger().info(
            "Starting persistent Gazebo adapter."
        )

        self.controller.wait_for_odom()

        x, y, yaw = self.controller.get_state()

        self.controller.get_logger().info(
            f"Initial pose: "
            f"x={x:.3f}, "
            f"y={y:.3f}, "
            f"yaw={math.degrees(yaw):.1f} deg"
        )

    def execute(self, action):
        """
        Execute one cardinal action:
        north, east, south, or west.
        """

        action = action.lower().strip()

        self.controller.execute_action(action)

        return self.get_state()

    def get_state(self):
        """
        Return the current continuous Gazebo pose.
        """

        x, y, yaw = self.controller.get_state()

        return {
            "x": x,
            "y": y,
            "yaw": yaw,
            "yaw_degrees": (
                math.degrees(yaw)
                if yaw is not None
                else None
            ),
        }

    def stop(self):
        self.controller.stop()

    def close(self):
        """
        Stop the robot and cleanly shut down the ROS node.
        """

        self.controller.stop()
        self.controller.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()