import math
import time

from gazebo_turtlebot.grid_action_controller import (
    CARDINAL_HEADINGS,
    DISTANCE_TOLERANCE,
)


class GazeboWaypointExecutor:
    """
    Execute grid actions by driving toward absolute Gazebo waypoints.

    Unlike repeated relative 0.50 m movements, this prevents small
    per-step odometry errors from accumulating across a long episode.
    """

    def __init__(
        self,
        adapter,
    ):
        self.adapter = adapter
        self.controller = adapter.controller

    def execute_to_waypoint(
        self,
        action,
        target_x,
        target_y,
    ):
        action = (
            action.lower().strip()
        )

        if action not in CARDINAL_HEADINGS:
            raise ValueError(
                f"Unknown action '{action}'."
            )

        before = (
            self.adapter.get_state()
        )

        dx = (
            target_x - before["x"]
        )

        dy = (
            target_y - before["y"]
        )

        distance = math.hypot(
            dx,
            dy,
        )

        self.controller.get_logger().info(
            "Absolute waypoint: "
            f"x={target_x:.3f}, "
            f"y={target_y:.3f}"
        )

        self.controller.get_logger().info(
            f"Current waypoint error: "
            f"{distance:.3f} m"
        )

        if distance > DISTANCE_TOLERANCE:
            travel_heading = math.atan2(
                dy,
                dx,
            )

            self.controller.rotate_to_heading(
                travel_heading
            )

            time.sleep(0.1)

            self.controller.drive_distance(
                distance
            )

            time.sleep(0.1)

        # Finish facing exactly as the logical cardinal action specifies.
        self.controller.rotate_to_heading(
            CARDINAL_HEADINGS[action]
        )

        time.sleep(0.1)

        self.controller.stop()

        return self.adapter.get_state()