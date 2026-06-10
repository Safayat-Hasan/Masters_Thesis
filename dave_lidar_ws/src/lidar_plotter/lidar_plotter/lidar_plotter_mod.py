import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
import math


class LidarPlotterMod(Node):

    def __init__(self):
        super().__init__('lidar_plotter_mod')

        self.subscription = self.create_subscription(
            LaserScan,
            '/lidar',
            self.listener_callback,
            10
        )

        self.saved = False
        self.get_logger().info("Lidar Plotter Node Started")

    def listener_callback(self, msg):

        if self.saved:
            return

        data = []

        for i, r in enumerate(msg.ranges):
            theta = msg.angle_min + i * msg.angle_increment

            if math.isfinite(r) and r < msg.range_max - 0.05:
                x = r * math.cos(theta)
                z = r * math.sin(theta)
                data.append([i, theta, math.degrees(theta), r, x, z])

        if len(data) == 0:
            self.get_logger().warn("No valid lidar points found.")
            return

        np.savetxt(
            "lidar_scan_points.csv",
            np.array(data),
            delimiter=",",
            header="beam_id,theta_rad,theta_deg,range_m,x_m,z_m",
            comments=""
        )

        self.get_logger().info(
            f"Saved {len(data)} lidar points to lidar_scan_points.csv"
        )

        self.saved = True


def main(args=None):
    rclpy.init(args=args)
    node = LidarPlotterMod()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()