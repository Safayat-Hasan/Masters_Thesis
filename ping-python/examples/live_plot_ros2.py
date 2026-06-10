#!/usr/bin/env python3

import struct
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2


class LivePlotPoolNode(Node):
    def __init__(self):
        super().__init__("live_plot_pool_node")

        self.topic = "/cerulean/points_accumulated"

        self.sub = self.create_subscription(
            PointCloud2,
            self.topic,
            self.callback,
            10
        )

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.fig.canvas.manager.set_window_title("Live Sonar Cross Section - Pool")
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

            xs.append(x)
            zs.append(z)

        if not xs:
            return

        self.ax.clear()
        self.ax.scatter(xs, zs, s=3)

        self.ax.set_xlabel("Forward distance (m)")
        self.ax.set_ylabel("Vertical distance (m)")
        self.ax.set_title("Live Sonar Cross Section - Pool")
        self.ax.set_aspect("equal", adjustable="box")

        # Adjust these if needed
        self.ax.set_xlim(0, 20)
        self.ax.set_ylim(-10, 10)

        self.ax.grid(True, alpha=0.3)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def main():
    rclpy.init()
    node = LivePlotPoolNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
