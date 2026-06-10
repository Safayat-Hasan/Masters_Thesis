#!/usr/bin/env python3

import struct

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2


class LiveDensityFanNode(Node):
    def __init__(self):
        super().__init__("live_density_fan_node")

        # accumulated usually looks richer than latest single ping
        self.topic = "/cerulean/points_accumulated"

        self.sub = self.create_subscription(
            PointCloud2,
            self.topic,
            self.callback,
            10
        )

        self.get_logger().info(f"Subscribed to {self.topic}")

        # Fixed display area for a pool-style rightward fan
        self.x_min = 0.0
        self.x_max = 20.0
        self.z_min = -10.0
        self.z_max = 10.0

        # image resolution
        self.nx = 300
        self.nz = 300

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.fig.canvas.manager.set_window_title("Live Density Fan")
        plt.show(block=False)

    def callback(self, msg: PointCloud2):
        xs = []
        zs = []

        data = msg.data
        step = msg.point_step

        for i in range(msg.width):
            offset = i * step
            x = struct.unpack_from("<f", data, offset + 0)[0]
            z = struct.unpack_from("<f", data, offset + 8)[0]

            if not np.isfinite(x) or not np.isfinite(z):
                continue

            if x < self.x_min or x > self.x_max or z < self.z_min or z > self.z_max:
                continue

            xs.append(x)
            zs.append(z)

        if not xs:
            return

        hist, xedges, zedges = np.histogram2d(
            xs,
            zs,
            bins=[self.nx, self.nz],
            range=[[self.x_min, self.x_max], [self.z_min, self.z_max]],
        )

        # smooth visually by taking log
        image = np.log1p(hist.T)

        self.ax.clear()
        self.ax.imshow(
            image,
            origin="lower",
            extent=[self.x_min, self.x_max, self.z_min, self.z_max],
            aspect="equal",
            interpolation="nearest",
        )

        self.ax.set_xlabel("Forward / range direction (m)")
        self.ax.set_ylabel("Fan spread direction (m)")
        self.ax.set_title(f"Live Density Fan | {self.topic} | {len(xs)} points")
        self.ax.grid(False)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def main():
    rclpy.init()
    node = LiveDensityFanNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
