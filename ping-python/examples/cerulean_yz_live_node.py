#!/usr/bin/env python3

import sys
from collections import deque
from typing import Deque, List, Tuple, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2

from brping import Surveyor240, definitions


XYZPoint = Tuple[float, float, float]


class CeruleanYZLiveNode(Node):
    def __init__(self) -> None:
        super().__init__("cerulean_yz_live_node")

        self.declare_parameter("sonar_ip", "192.168.2.86")
        self.declare_parameter("sonar_port", 62312)
        self.declare_parameter("frame_id", "cerulean_sonar_link")
        self.declare_parameter("start_mm", 0)
        self.declare_parameter("end_mm", 20000)
        self.declare_parameter("buffer_pings", 20)

        self.sonar_ip: str = self.get_parameter("sonar_ip").value
        self.sonar_port: int = int(self.get_parameter("sonar_port").value)
        self.frame_id: str = self.get_parameter("frame_id").value
        self.start_mm: int = int(self.get_parameter("start_mm").value)
        self.end_mm: int = int(self.get_parameter("end_mm").value)
        self.buffer_pings: int = int(self.get_parameter("buffer_pings").value)

        self.pub_latest = self.create_publisher(PointCloud2, "/cerulean/points", 10)
        self.pub_accum = self.create_publisher(PointCloud2, "/cerulean/points_accumulated", 10)

        self.ping_buffer: Deque[List[XYZPoint]] = deque(maxlen=self.buffer_pings)

        self.sonar = Surveyor240()
        self.connected = False

        self.get_logger().info(f"Connecting to Surveyor240 at {self.sonar_ip}:{self.sonar_port}")
        self._connect_and_start()

        self.timer = self.create_timer(0.02, self.poll_sonar)

    def _connect_and_start(self) -> None:
        self.sonar.connect_udp(self.sonar_ip, self.sonar_port)

        if not self.sonar.initialize():
            raise RuntimeError("Failed to initialize Surveyor240")

        self.get_logger().info("Surveyor240 initialized")

        self.sonar.control_set_ping_parameters(
            start_mm=self.start_mm,
            end_mm=self.end_mm,
            ping_enable=True,
            enable_yz_point_data=True,
            enable_atof_data=False,
        )

        self.connected = True
        self.get_logger().info(
            f"Pinging enabled | YZ mode | range {self.start_mm} mm to {self.end_mm} mm"
        )

    def poll_sonar(self) -> None:
        if not self.connected:
            return

        try:
            data = self.sonar.wait_message([
                definitions.SURVEYOR240_YZ_POINT_DATA,
                definitions.SURVEYOR240_ATTITUDE_REPORT,
                definitions.SURVEYOR240_WATER_STATS,
            ])

            if not data:
                return

            if data.message_id == definitions.SURVEYOR240_YZ_POINT_DATA:
                self.handle_yz(data)

        except Exception as e:
            self.get_logger().error(f"Polling error: {e}")

    def decode_yz_points(self, data) -> List[Tuple[float, float]]:
        """
        Tries to decode YZ point data robustly, because library versions differ.
        Returns a list of (y, z) tuples in sonar local slice coordinates.
        """
        yz = Surveyor240.create_yz_point_data(data)

        # Case 1: list/tuple of objects with .y and .z
        if len(yz) > 0 and hasattr(yz[0], "y") and hasattr(yz[0], "z"):
            return [(float(p.y), float(p.z)) for p in yz]

        # Case 2: list/tuple of 2-tuples
        if len(yz) > 0 and isinstance(yz[0], (list, tuple)) and len(yz[0]) >= 2:
            return [(float(p[0]), float(p[1])) for p in yz]

        # Case 3: flat list [y0, z0, y1, z1, ...]
        if isinstance(yz, (list, tuple)) and len(yz) % 2 == 0:
            pts = []
            for i in range(0, len(yz), 2):
                pts.append((float(yz[i]), float(yz[i + 1])))
            return pts

        raise RuntimeError(f"Unsupported YZ packet format: type={type(yz)}, sample={yz[:10] if hasattr(yz, '__getitem__') else yz}")

    def handle_yz(self, data) -> None:
        try:
            yz_points = self.decode_yz_points(data)
        except Exception as e:
            self.get_logger().error(f"Failed to decode YZ packet: {e}")
            return

        latest_points: List[XYZPoint] = []

        for y_sonar, z_sonar in yz_points:
            # Map sonar slice to RViz cross-section:
            # x = horizontal distance
            # y = 0 so the slice stays flat
            # z = vertical
            #
            # The minus sign flips horizontal direction to look more like your SonarView CSV.
            x = -y_sonar
            y = 0.0
            z = z_sonar

            # basic sanity filter
            if abs(x) > 50 or abs(z) > 50:
                continue

            latest_points.append((x, y, z))

        if not latest_points:
            return

        self.ping_buffer.append(latest_points)

        accum_points: List[XYZPoint] = []
        for ping in self.ping_buffer:
            accum_points.extend(ping)

        self.pub_latest.publish(self.make_cloud(latest_points))
        self.pub_accum.publish(self.make_cloud(accum_points))

        self.get_logger().info(
            f"Latest: {len(latest_points)} pts | Accumulated: {len(accum_points)} pts",
            throttle_duration_sec=2.0
        )

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
    node: Optional[CeruleanYZLiveNode] = None

    try:
        node = CeruleanYZLiveNode()
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
