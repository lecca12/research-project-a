import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


WAYPOINTS = [
    (1.0, 0.0),
    (1.0, 1.0),
    (0.0, 1.0),
    (0.0, 0.0),
]

LINEAR_SPEED = 0.15
ANGULAR_SPEED = 0.5

DISTANCE_TOLERANCE = 0.08
ANGLE_TOLERANCE = 0.08


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi

    while angle < -math.pi:
        angle += 2.0 * math.pi

    return angle


class WaypointDriver(Node):
    def __init__(self):
        super().__init__("waypoint_driver")

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

        self.timer = self.create_timer(
            0.05,
            self.control_loop,
        )

        self.x = None
        self.y = None
        self.yaw = None

        self.current_waypoint_index = 0
        self.finished = False

        self.get_logger().info(
            f"Waypoint driver started with {len(WAYPOINTS)} waypoints."
        )

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation

        siny_cosp = 2.0 * (
            q.w * q.z
            + q.x * q.y
        )

        cosy_cosp = 1.0 - 2.0 * (
            q.y * q.y
            + q.z * q.z
        )

        self.yaw = math.atan2(
            siny_cosp,
            cosy_cosp,
        )

    def publish_stop(self):
        msg = Twist()
        self.cmd_pub.publish(msg)

    def control_loop(self):
        if self.finished:
            self.publish_stop()
            return

        if (
            self.x is None
            or self.y is None
            or self.yaw is None
        ):
            return

        if self.current_waypoint_index >= len(WAYPOINTS):
            self.get_logger().info(
                "All waypoints reached."
            )

            self.publish_stop()
            self.finished = True
            return

        target_x, target_y = WAYPOINTS[
            self.current_waypoint_index
        ]

        dx = target_x - self.x
        dy = target_y - self.y

        distance = math.hypot(
            dx,
            dy,
        )

        if distance < DISTANCE_TOLERANCE:
            self.get_logger().info(
                f"Reached waypoint "
                f"{self.current_waypoint_index + 1}: "
                f"({target_x:.2f}, {target_y:.2f})"
            )

            self.current_waypoint_index += 1
            self.publish_stop()
            return

        desired_yaw = math.atan2(
            dy,
            dx,
        )

        yaw_error = normalize_angle(
            desired_yaw - self.yaw
        )

        cmd = Twist()

        if abs(yaw_error) > ANGLE_TOLERANCE:
            cmd.linear.x = 0.0

            cmd.angular.z = (
                ANGULAR_SPEED
                if yaw_error > 0
                else -ANGULAR_SPEED
            )

        else:
            cmd.linear.x = LINEAR_SPEED
            cmd.angular.z = 0.0

        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)

    node = WaypointDriver()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.publish_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
