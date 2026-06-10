#!/usr/bin/env python3

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist
from tf2_msgs.msg import TFMessage


class LidarOnlineMappingNode(Node):
    def __init__(self):
        super().__init__("lidar_online_mapping_node")

        self.resolution = 0.1
        self.width = 200
        self.height = 200
        self.origin_x = -10.0
        self.origin_y = -10.0

        self.grid = np.zeros((self.height, self.width), dtype=np.float32)

        self.pose_valid = False
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0

        self.create_subscription(
            LaserScan,
            "/simple_lidar_bot/lidar",
            self.scan_callback,
            qos_profile_sensor_data
        )

        self.create_subscription(
            TFMessage,
            "/world/gpu_lidar_sensor/dynamic_pose/info",
            self.pose_callback,
            20,
        )

        self.map_pub = self.create_publisher(OccupancyGrid, "/map", 10)
        #self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self.get_logger().info("Lidar online mapping node started.")

    def pose_callback(self, msg: TFMessage):
        chosen = None

        for tf in msg.transforms:
            tx = tf.transform.translation.x
            ty = tf.transform.translation.y
            tz = tf.transform.translation.z

            if abs(tx) > 1e-6 or abs(ty) > 1e-6 or abs(tz) > 1e-6:
                chosen = tf
                break

        if chosen is None:
            self.get_logger().warn("Pose callback got only zero transforms.", throttle_duration_sec=2.0)
            return

        self.robot_x = chosen.transform.translation.x
        self.robot_y = chosen.transform.translation.y

        q = chosen.transform.rotation
        self.robot_yaw = self.yaw_from_quaternion(q.x, q.y, q.z, q.w)

        self.pose_valid = True

        self.get_logger().info(
            f"Pose: x={self.robot_x:.2f}, y={self.robot_y:.2f}, yaw={self.robot_yaw:.2f}",
            throttle_duration_sec=1.0
        )

    def scan_callback(self, msg: LaserScan):
        self.get_logger().info(
            f"Received scan with {len(msg.ranges)} ranges",
            throttle_duration_sec=1.0
        )

        if not self.pose_valid:
            self.get_logger().warn("No valid pose yet, skipping scan.", throttle_duration_sec=2.0)
            return

        #self.grid *= 0.995
        self.grid *= 0.98

        front_min = float("inf")
        valid_points = 0

        for i, r in enumerate(msg.ranges):
            if not math.isfinite(r):
                continue
            if r < msg.range_min or r > msg.range_max:
                continue

            angle = msg.angle_min + i * msg.angle_increment

            x_local = r * math.cos(angle)
            y_local = r * math.sin(angle)

            x_world = (
                self.robot_x
                + x_local * math.cos(self.robot_yaw)
                - y_local * math.sin(self.robot_yaw)
            )
            y_world = (
                self.robot_y
                + x_local * math.sin(self.robot_yaw)
                + y_local * math.cos(self.robot_yaw)
            )

            self.mark_ray_free(self.robot_x, self.robot_y, x_world, y_world)
            self.mark_occupied(x_world, y_world)

            if abs(angle) < math.radians(20):
                front_min = min(front_min, r)

            valid_points += 1

        self.get_logger().info(
            f"Processed {valid_points} valid lidar points",
            throttle_duration_sec=1.0
        )

        self.publish_map()
        #self.simple_avoidance(front_min)

    def yaw_from_quaternion(self, x, y, z, w):
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def world_to_grid(self, x, y):
        gx = int((x - self.origin_x) / self.resolution)
        gy = int((y - self.origin_y) / self.resolution)
        return gx, gy

    def mark_occupied(self, x, y):
        gx, gy = self.world_to_grid(x, y)
        if 0 <= gx < self.width and 0 <= gy < self.height:
            self.grid[gy, gx] = min(self.grid[gy, gx] + 8.0, 100.0)

    def mark_free(self, x, y):
        gx, gy = self.world_to_grid(x, y)
        if 0 <= gx < self.width and 0 <= gy < self.height:
            self.grid[gy, gx] = max(self.grid[gy, gx] - 2.0, 0.0)

    def mark_ray_free(self, x0, y0, x1, y1):
        dist = math.hypot(x1 - x0, y1 - y0)
        steps = max(1, int(dist / self.resolution))

        for k in range(steps):
            t = k / steps
            xr = x0 + t * (x1 - x0)
            yr = y0 + t * (y1 - y0)
            self.mark_free(xr, yr)

    def publish_map(self):
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        msg.info.resolution = self.resolution
        msg.info.width = self.width
        msg.info.height = self.height
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0

        out = np.clip(self.grid, 0, 100).astype(np.int8)
        msg.data = out.flatten().tolist()

        self.map_pub.publish(msg)
        self.get_logger().info("Published /map", throttle_duration_sec=1.0)

    def simple_avoidance(self, front_min):
        cmd = Twist()

        if not math.isfinite(front_min):
            cmd.linear.x = 0.3
        elif front_min < 1.5:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.6
        elif front_min < 3.0:
            cmd.linear.x = 0.1
            cmd.angular.z = 0.3
        else:
            cmd.linear.x = 0.4
            cmd.angular.z = 0.0

        self.cmd_pub.publish(cmd)


def main():
    rclpy.init()
    node = LidarOnlineMappingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
