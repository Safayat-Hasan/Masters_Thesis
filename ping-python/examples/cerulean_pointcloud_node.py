#!/usr/bin/env python3

import math
import struct
import signal
import sys
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2

from brping import Surveyor240, definitions


class CeruleanPointCloudNode(Node):
    def __init__(self):
        super().__init__("cerulean_pointcloud_node")

        # Parameters
        self.declare_parameter("sonar_ip", "192.168.2.86")
        self.declare_parameter("sonar_port", 62312)
        self.declare_parameter("frame_id", "cerulean_sonar_link")
        self.declare_parameter("speed_of_sound", 1500.0)   # m/s
        self.declare_parameter("start_mm", 0)
        self.declare_parameter("end_mm", 5000)
        self.declare_parameter("publish_rate_hz", 10.0)

        self.sonar_ip = self.get_parameter("sonar_ip").value
        self.sonar_port = int(self.get_parameter("sonar_port").value)
        self.frame_id = self.get_parameter("frame_id").value
        self.speed_of_sound = float(self.get_parameter("speed_of_sound").value)
        self.start_mm = int(self.get_parameter("start_mm").value)
        self.end_mm = int(self.get_parameter("end_mm").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)

        self.publisher_ = self.create_publisher(PointCloud2, "/cerulean/points", 10)

        self.sonar = Surveyor240()
        self.connected = False

        self.get_logger().info(
            f"Connecting to Surveyor240 at {self.sonar_ip}:{self.sonar_port}"
        )
        self._connect_and_start()

        timer_period = 1.0 / self.publish_rate_hz
        self.timer = self.create_timer(timer_period, self.poll_sonar)

    def _connect_and_start(self) -> None:
        try:
            self.sonar.connect_udp(self.sonar_ip, self.sonar_port)

            if not self.sonar.initialize():
                raise RuntimeError("Failed to initialize Surveyor240")

            self.get_logger().info("Surveyor240 initialized")

            # Enable pinging and ATOF data.
            # YZ disabled here because we want to compute XYZ ourselves from ATOF.
            self.sonar.control_set_ping_parameters(
                start_mm=self.start_mm,
                end_mm=self.end_mm,
                ping_enable=True,
                enable_yz_point_data=False,
                enable_atof_data=True,
            )

            self.connected = True
            self.get_logger().info("Pinging enabled")
        except Exception as e:
            self.connected = False
            self.get_logger().error(f"Connection/start failed: {e}")
            raise

    def poll_sonar(self) -> None:
        if not self.connected:
            return

        try:
            data = self.sonar.wait_message(
                [
                    definitions.SURVEYOR240_ATOF_POINT_DATA,
                    definitions.SURVEYOR240_ATTITUDE_REPORT,
                    definitions.SURVEYOR240_WATER_STATS,
                ]
            )

            if not data:
                return

            if data.message_id == definitions.SURVEYOR240_ATTITUDE_REPORT:
                # Optional: log slowly if you want.
                return

            if data.message_id == definitions.SURVEYOR240_WATER_STATS:
                # Optional: update speed of sound from sensor if desired.
                return

            if data.message_id == definitions.SURVEYOR240_ATOF_POINT_DATA:
                self.handle_atof(data)

        except Exception as e:
            self.get_logger().error(f"Error while polling sonar: {e}")

    def handle_atof(self, data) -> None:
        try:
            atof_points = Surveyor240.create_atof_list(data)
        except Exception as e:
            self.get_logger().error(f"Failed to decode ATOF packet: {e}")
            return

        xyz_points: List[Tuple[float, float, float]] = []

        for p in atof_points:
            try:
                angle = float(p.angle)
                tof = float(p.tof)

                # Range from time-of-flight
                distance = 0.5 * self.speed_of_sound * tof

                # Local sonar frame.
                # Assumption:
                #   x = 0 in the along-track direction for one ping slice
                #   y = across-track
                #   z = down/up signed by beam convention
                #
                # Adjust signs later if your RViz view looks mirrored.
                x = 0.0
                y = distance * math.sin(angle)
                z = -distance * math.cos(angle)

                xyz_points.append((x, y, z))
            except Exception as e:
                self.get_logger().warn(f"Skipping bad point: {e}")

        if not xyz_points:
            return

        cloud_msg = self.create_pointcloud2(xyz_points)
        self.publisher_.publish(cloud_msg)

        self.get_logger().info(
            f"Published {len(xyz_points)} points",
            throttle_duration_sec=2.0
        )

    def create_pointcloud2(self, points: List[Tuple[float, float, float]]) -> PointCloud2:
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.frame_id

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        return point_cloud2.create_cloud(header, fields, points)

    def stop_sonar(self) -> None:
        try:
            if self.connected:
                self.get_logger().info("Stopping pinging...")
                self.sonar.control_set_ping_parameters(ping_enable=False)
        except Exception as e:
            self.get_logger().warn(f"Failed to stop pinging cleanly: {e}")

        try:
            if getattr(self.sonar, "iodev", None):
                self.sonar.iodev.close()
        except Exception as e:
            self.get_logger().warn(f"Failed to close socket cleanly: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = CeruleanPointCloudNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
    finally:
        if node is not None:
            node.stop_sonar()
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
