import argparse
import math
import threading
import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


CELL_SIZE_METRES = 0.50

LINEAR_SPEED = 0.15
ANGULAR_SPEED = 0.50

DISTANCE_TOLERANCE = 0.03
ANGLE_TOLERANCE = 0.04

CONTROL_PERIOD = 0.05


CARDINAL_HEADINGS = {
    "east": 0.0,
    "north": math.pi / 2.0,
    "west": math.pi,
    "south": -math.pi / 2.0,
}


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi

    while angle < -math.pi:
        angle += 2.0 * math.pi

    return angle


class GridActionController(Node):
    def __init__(self):
        super().__init__("grid_action_controller")

        self.cmd_pub = self.create_publisher(
            Twist,
            "/cmd_vel",
            10,
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            10,
        )

        self.x = None
        self.y = None
        self.yaw = None

        self.state_lock = threading.Lock()

        self.get_logger().info(
            "Grid action controller started."
        )

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation

        siny_cosp = 2.0 * (
            q.w * q.z
            + q.x * q.y
        )

        cosy_cosp = 1.0 - 2.0 * (
            q.y * q.y
            + q.z * q.z
        )

        yaw = math.atan2(
            siny_cosp,
            cosy_cosp,
        )

        with self.state_lock:
            self.x = x
            self.y = y
            self.yaw = yaw

    def get_state(self):
        with self.state_lock:
            return self.x, self.y, self.yaw

    def stop(self):
        msg = Twist()
        self.cmd_pub.publish(msg)

    def wait_for_odom(self):
        self.get_logger().info(
            "Waiting for /odom..."
        )

        while rclpy.ok():
            x, y, yaw = self.get_state()

            if (
                x is not None
                and y is not None
                and yaw is not None
            ):
                self.get_logger().info(
                    "Odometry received."
                )
                return

            rclpy.spin_once(
                self,
                timeout_sec=0.1,
            )

    def rotate_to_heading(self, target_yaw):
        self.get_logger().info(
            f"Rotating to heading "
            f"{math.degrees(target_yaw):.1f} degrees."
        )

        while rclpy.ok():
            rclpy.spin_once(
                self,
                timeout_sec=CONTROL_PERIOD,
            )

            _, _, current_yaw = self.get_state()

            if current_yaw is None:
                continue

            error = normalize_angle(
                target_yaw - current_yaw
            )

            if abs(error) <= ANGLE_TOLERANCE:
                self.stop()

                self.get_logger().info(
                    "Heading reached."
                )
                return

            msg = Twist()

            msg.angular.z = (
                ANGULAR_SPEED
                if error > 0
                else -ANGULAR_SPEED
            )

            self.cmd_pub.publish(msg)

    def drive_distance(self, distance):
        start_x, start_y, _ = self.get_state()

        if start_x is None or start_y is None:
            raise RuntimeError(
                "No odometry available before drive."
            )

        self.get_logger().info(
            f"Driving {distance:.2f} m."
        )

        while rclpy.ok():
            rclpy.spin_once(
                self,
                timeout_sec=CONTROL_PERIOD,
            )

            current_x, current_y, _ = self.get_state()

            if current_x is None or current_y is None:
                continue

            travelled = math.hypot(
                current_x - start_x,
                current_y - start_y,
            )

            remaining = distance - travelled

            if remaining <= DISTANCE_TOLERANCE:
                self.stop()

                self.get_logger().info(
                    f"Distance reached. "
                    f"Travelled={travelled:.3f} m"
                )
                return

            msg = Twist()
            msg.linear.x = LINEAR_SPEED

            self.cmd_pub.publish(msg)

    def execute_action(self, action):
        action = action.lower().strip()

        if action not in CARDINAL_HEADINGS:
            raise ValueError(
                f"Unknown action '{action}'. "
                "Expected north, east, south, or west."
            )

        target_heading = CARDINAL_HEADINGS[action]

        start_x, start_y, start_yaw = self.get_state()

        self.get_logger().info(
            f"Executing action: {action}"
        )

        self.get_logger().info(
            f"Start pose: "
            f"x={start_x:.3f}, "
            f"y={start_y:.3f}, "
            f"yaw={math.degrees(start_yaw):.1f} deg"
        )

        self.rotate_to_heading(
            target_heading
        )

        time.sleep(0.2)

        self.drive_distance(
            CELL_SIZE_METRES
        )

        time.sleep(0.2)

        end_x, end_y, end_yaw = self.get_state()

        self.stop()

        self.get_logger().info(
            f"Completed action: {action}"
        )

        self.get_logger().info(
            f"End pose: "
            f"x={end_x:.3f}, "
            f"y={end_y:.3f}, "
            f"yaw={math.degrees(end_yaw):.1f} deg"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Execute one cardinal grid action "
            "with a TurtleBot3 in Gazebo."
        )
    )

    parser.add_argument(
        "action",
        choices=[
            "north",
            "east",
            "south",
            "west",
        ],
        help=(
            "Cardinal action to execute."
        ),
    )

    return parser.parse_args()


def main(args=None):
    cli_args = parse_args()

    rclpy.init(args=args)

    node = GridActionController()

    try:
        node.wait_for_odom()

        node.execute_action(
            cli_args.action
        )

    except KeyboardInterrupt:
        node.get_logger().info(
            "Interrupted."
        )

    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
