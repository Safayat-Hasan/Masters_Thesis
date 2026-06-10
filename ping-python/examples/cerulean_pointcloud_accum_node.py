#!/usr/bin/env python3

import math
import sys
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2

from brping import Surveyor240, definitions


XYZPoint = Tuple[float, float, float]


class CeruleanAccumPointCloudNode(Node):
    def __init__(self) -> None:
        super().__init__("cerulean_accum_pointcloud_node")

        # Connection / sonar parameters
        self.declare_parameter("sonar_ip", "192.168.2.86")
        self.declare_parameter("sonar_port", 62312)
        self.declare_parameter("frame_id", "cerulean_sonar_link")
        self.declare_parameter("speed_of_sound", 1500.0)   # m/s fallback
        self.declare_parameter("start_mm", 0)
        self.declare_parameter("end_mm", 20000)            # 20 m default for pool use
        self.declare_parameter("publish_rate_hz", 10.0)

        # Accumulation / filtering parameters
        self.declare_parameter("buffer_pings", 150)
        self.declare_parameter("min_range_m", 0.1)
        self.declare_parameter("max_range_m", 25.0)

        # Optional logging
        self.declare_parameter("save_csv", False)
        self.declare_parameter("csv_path", "/tmp/cerulean_live_accum_points.csv")

        self.sonar_ip: str = self.get_parameter("sonar_ip").value
        self.sonar_port: int = int(self.get_parameter("sonar_port").value)
        self.frame_id: str = self.get_parameter("frame_id").value
        self.speed_of_sound: float = float(self.get_parameter("speed_of_sound").value)
        self.start_mm: int = int(self.get_parameter("start_mm").value)
        self.end_mm: int = int(self.get_parameter("end_mm").value)
        self.publish_rate_hz: float = float(self.get_parameter("publish_rate_hz").value)

        self.buffer_pings: int = int(self.get_parameter("buffer_pings").value)
        self.min_range_m: float = float(self.get_parameter("min_range_m").value)
        self.max_range_m: float = float(self.get_parameter("max_range_m").value)

        self.save_csv: bool = bool(self.get_parameter("save_csv").value)
        self.csv_path: str = self.get_parameter("csv_path").value

        self.pub_latest = self.create_publisher(PointCloud2, "/cerulean/points", 10)
        self.pub_accum = self.create_publisher(PointCloud2, "/cerulean/points_accumulated", 10)

        self.sonar = Surveyor240()
        self.connected = False

        self.ping_buffer: Deque[List[XYZPoint]] = deque(maxlen=self.buffer_pings)
        self.total_pings_received = 0
        self.total_points_received = 0

        if self.save_csv:
            self._prepare_csv()

        self.get_logger().info(
            f"Connecting to Surveyor240 at {self.sonar_ip}:{self.sonar_port}"
        )
        self._connect_and_start()

        timer_period = 1.0 / self.publish_rate_hz
        self.timer = self.create_timer(timer_period, self.poll_sonar)

    def _prepare_csv(self) -> None:
        csv_file = Path(self.csv_path)
        csv_file.parent.mkdir(parents=True, exist_ok=True)
        if not csv_file.exists():
            with open(csv_file, "w", encoding="utf-8") as f:
                f.write("x,y,z\n")
        self.get_logger().info(f"CSV logging enabled: {self.csv_path}")

    def _connect_and_start(self) -> None:
        try:
            self.sonar.connect_udp(self.sonar_ip, self.sonar_port)

            if not self.sonar.initialize():
                raise RuntimeError("Failed to initialize Surveyor240")

            self.get_logger().info("Surveyor240 initialized")

            self.sonar.control_set_ping_parameters(
                start_mm=self.start_mm,
                end_mm=self.end_mm,
                ping_enable=True,
                enable_yz_point_data=False,
                enable_atof_data=True,
            )

            self.connected = True
            self.get_logger().info(
                f"Pinging enabled | range: {self.start_mm} mm to {self.end_mm} mm | "
                f"buffer_pings: {self.buffer_pings}"
            )

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

            if data.message_id == definitions.SURVEYOR240_ATOF_POINT_DATA:
                self.handle_atof(data)

            elif data.message_id == definitions.SURVEYOR240_ATTITUDE_REPORT:
                # Keeping quiet here to avoid spam.
                return

            elif data.message_id == definitions.SURVEYOR240_WATER_STATS:
                # Optional future upgrade:
                # update self.speed_of_sound from packet if desired.
                return

        except Exception as e:
            self.get_logger().error(f"Polling error: {e}")

    def handle_atof(self, data) -> None:
        try:
            atof_points = Surveyor240.create_atof_list(data)
        except Exception as e:
            self.get_logger().error(f"Failed to decode ATOF packet: {e}")
            return

        latest_points: List[XYZPoint] = []

        for p in atof_points:
            try:
                angle = float(p.angle)
                tof = float(p.tof)

                distance = 0.5 * self.speed_of_sound * tof

                if distance < self.min_range_m or distance > self.max_range_m:
                    continue

                # Single ping slice in local sonar frame.
                # This assumes the sonar is fixed while accumulating.
                #
                # x: along-track (set to 0 for one slice)
                # y: cross-track
                # z: vertical
                x = distance * math.cos(angle)
                y = 0.0
                z = distance * math.sin(angle)
                latest_points.append((x, y, z))

            except Exception as e:
                self.get_logger().warn(f"Skipping bad point: {e}")

        if not latest_points:
            self.get_logger().warn(
                "Received ATOF packet but no valid points after filtering.",
                throttle_duration_sec=2.0
            )
            return

        self.total_pings_received += 1
        self.total_points_received += len(latest_points)

        self.ping_buffer.append(latest_points)

        accum_points: List[XYZPoint] = []
        for ping in self.ping_buffer:
            accum_points.extend(ping)

        latest_msg = self.make_cloud(latest_points)
        accum_msg = self.make_cloud(accum_points)

        self.pub_latest.publish(latest_msg)
        self.pub_accum.publish(accum_msg)

        if self.save_csv:
            self._append_points_to_csv(latest_points)

        self.get_logger().info(
            f"Ping #{self.total_pings_received} | latest: {len(latest_points)} pts | "
            f"accumulated: {len(accum_points)} pts",
            throttle_duration_sec=2.0
        )

    def _append_points_to_csv(self, points: List[XYZPoint]) -> None:
        try:
            with open(self.csv_path, "a", encoding="utf-8") as f:
                for x, y, z in points:
                    f.write(f"{x},{y},{z}\n")
        except Exception as e:
            self.get_logger().warn(f"Failed to append points to CSV: {e}")

    def make_cloud(self, points: List[XYZPoint]) -> PointCloud2:
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


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[CeruleanAccumPointCloudNode] = None

    try:
        node = CeruleanAccumPointCloudNode()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)

    finally:
        if node is not None:
            node.stop_sonar()
            node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
