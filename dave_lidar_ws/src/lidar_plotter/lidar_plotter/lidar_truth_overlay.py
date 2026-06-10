import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class LidarTruthOverlay(Node):
    def __init__(self):
        super().__init__('lidar_truth_overlay')

        self.subscription = self.create_subscription(
            LaserScan,
            '/lidar',
            self.scan_callback,
            10
        )

        # -----------------------------
        # Ground-truth values from Gazebo
        # -----------------------------

        # Actual lidar pose in WORLD frame
        # model_with_lidar pose + rotated local link offset (0.05, 0.05, 0.05)
        self.lidar_pos_world = np.array([3.95, -0.05, 0.55])

        # model_with_lidar quaternion [x, y, z, w]
        self.lidar_quat_world = np.array([
            0.0,
            0.0,
            0.99999968293183461,
            0.00079632671073326324
        ])

        # Box pose in WORLD frame
        self.box_pos_world = np.array([
            -0.067625917781690165,
            -0.92842914779487173,
             0.54999999632499452
        ])

        # Box quaternion [x, y, z, w]
        self.box_quat_world = np.array([
            -3.4664925160894867e-18,
             3.3882787136626588e-18,
             0.0013190257935218094,
             0.99999913008509966
        ])

        # Box dimensions from SDF
        self.box_size = np.array([1.0, 1.0, 1.0])

        # Precompute truth geometry in lidar frame
        self.truth_corners_sensor = self.compute_box_corners_in_lidar_frame()
        self.truth_profile_xz = self.compute_box_scan_plane_intersection()

        # Plot setup
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 6))

        self.get_logger().info('Lidar truth overlay node started.')

    def transform_world_to_sensor(self, points_world, sensor_pos_world, sensor_quat_world):
        """Transform Nx3 world points into sensor frame."""
        R_ws = R.from_quat(sensor_quat_world).as_matrix()
        points_rel = points_world - sensor_pos_world
        points_sensor = (R_ws.T @ points_rel.T).T
        return points_sensor

    def box_corners_world(self, box_pos, box_size, box_quat):
        """Return 8 box corners in world frame."""
        lx, ly, lz = box_size
        dx, dy, dz = lx / 2.0, ly / 2.0, lz / 2.0

        corners_local = np.array([
            [-dx, -dy, -dz],
            [-dx, -dy,  dz],
            [-dx,  dy, -dz],
            [-dx,  dy,  dz],
            [ dx, -dy, -dz],
            [ dx, -dy,  dz],
            [ dx,  dy, -dz],
            [ dx,  dy,  dz],
        ])

        R_box = R.from_quat(box_quat).as_matrix()
        corners_world = (R_box @ corners_local.T).T + box_pos
        return corners_world

    def compute_box_corners_in_lidar_frame(self):
        corners_world = self.box_corners_world(
            self.box_pos_world,
            self.box_size,
            self.box_quat_world
        )
        corners_sensor = self.transform_world_to_sensor(
            corners_world,
            self.lidar_pos_world,
            self.lidar_quat_world
        )
        return corners_sensor

    def compute_box_scan_plane_intersection(self):
        """
        Compute the intersection of the box with the lidar 2D scan plane.
        The lidar plot is in sensor frame x-z, so the scan plane is y = 0.

        We transform the box corners to sensor frame, then find where box edges
        cross y=0. Those intersections form the true 2D profile seen by the lidar.
        """
        corners = self.truth_corners_sensor

        # 12 box edges by corner index
        edges = [
            (0, 1), (0, 2), (0, 4),
            (1, 3), (1, 5),
            (2, 3), (2, 6),
            (3, 7),
            (4, 5), (4, 6),
            (5, 7),
            (6, 7),
        ]

        intersections = []

        for i, j in edges:
            p1 = corners[i]
            p2 = corners[j]

            y1 = p1[1]
            y2 = p2[1]

            # If both exactly on plane, keep both endpoints
            if abs(y1) < 1e-9 and abs(y2) < 1e-9:
                intersections.append(p1)
                intersections.append(p2)
                continue

            # If the edge crosses y=0
            if y1 * y2 < 0.0:
                t = -y1 / (y2 - y1)
                p = p1 + t * (p2 - p1)
                intersections.append(p)

            # If one endpoint lies exactly on the plane
            elif abs(y1) < 1e-9:
                intersections.append(p1)
            elif abs(y2) < 1e-9:
                intersections.append(p2)

        if len(intersections) == 0:
            return np.empty((0, 3))

        pts = np.array(intersections)

        # Remove duplicates approximately
        rounded = np.round(pts, decimals=6)
        _, idx = np.unique(rounded, axis=0, return_index=True)
        pts = pts[np.sort(idx)]

        return pts

    def scan_callback(self, msg):
        ranges = msg.ranges
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment

        x_points = []
        z_points = []

        for i, r in enumerate(ranges):
            if math.isfinite(r):
                theta = angle_min + i * angle_increment

                # Since the scan is in lidar x-z plane:
                x = r * math.cos(theta)
                z = r * math.sin(theta)

                x_points.append(x)
                z_points.append(z)

        x_points = np.array(x_points)
        z_points = np.array(z_points)

        self.ax.clear()

        # Plot reconstructed map
        if len(x_points) > 0:
            self.ax.scatter(
                x_points, z_points,
                s=8, c='blue', label='Reconstructed lidar map'
            )

        # Plot ground-truth box corners in lidar frame
        corners = self.truth_corners_sensor
        self.ax.scatter(
            corners[:, 0], corners[:, 2],
            s=40, c='red', marker='x', label='Ground-truth box corners'
        )

        # Plot true scan-plane intersection if available
        profile = self.truth_profile_xz
        if len(profile) > 0:
            # Sort roughly by z for clean line drawing
            order = np.argsort(profile[:, 2])
            profile_sorted = profile[order]

            self.ax.plot(
                profile_sorted[:, 0], profile_sorted[:, 2],
                c='green', linewidth=2, label='Ground-truth box profile in scan plane'
            )
            self.ax.scatter(
                profile_sorted[:, 0], profile_sorted[:, 2],
                c='green', s=30
            )

        self.ax.set_xlabel('x in lidar frame (forward) [m]')
        self.ax.set_ylabel('z in lidar frame (vertical) [m]')
        self.ax.set_title('2D Reconstructed Map vs Gazebo Ground Truth')
        self.ax.grid(True)
        self.ax.axis('equal')
        self.ax.legend()
        self.ax.set_xlim(0, 6)
        self.ax.set_ylim(-3, 3)

        plt.draw()
        plt.pause(0.001)


def main(args=None):
    rclpy.init(args=args)
    node = LidarTruthOverlay()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
