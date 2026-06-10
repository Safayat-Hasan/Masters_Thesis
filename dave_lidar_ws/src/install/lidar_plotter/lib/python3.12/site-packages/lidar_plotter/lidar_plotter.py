import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
import matplotlib.pyplot as plt
import math

class LidarPlotter(Node):

    def __init__(self):
        super().__init__('lidar_plotter')

        self.subscription = self.create_subscription(
            LaserScan,
            '/lidar',
            self.listener_callback,
            10)

        self.subscription  # prevent unused warning

        # Setup matplotlib
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.scatter = None

        self.get_logger().info("Lidar Plotter Node Started")

    def listener_callback(self, msg):

        ranges = msg.ranges
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment

        x_points = []
        z_points = []

        for i, r in enumerate(ranges):
            if math.isfinite(r):
                theta = angle_min + i * angle_increment

                # Convert to X-Z plane
                x = r * math.cos(theta)
                z = r * math.sin(theta)

                x_points.append(x)
                z_points.append(z)

        # Clear previous plot
        self.ax.clear()

        # Plot points
        self.ax.scatter(x_points, z_points, s=5)

        self.ax.set_xlabel("Forward (x)")
        self.ax.set_ylabel("Vertical (z)")
        self.ax.set_title("2D Obstacle Map (X-Z Plane)")
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(-5, 5)
        self.ax.grid(True)
        self.ax.set_aspect('equal', adjustable='box')

        plt.draw()
        plt.pause(0.001)


def main(args=None):
    rclpy.init(args=args)
    node = LidarPlotter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
