import subprocess
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from scipy.spatial.transform import Rotation as R


# =========================
# INPUT FILES
# =========================

POSE_CSV = "blueboat_local_position.csv"
ATT_CSV = "blueboat_attitude.csv"
REAL_SONAR_CSV = "surveyor_atof.csv"

# Choose one ping. Use None to auto-pick one with many returns.
# PING_NUMBER = None
PING_NUMBER = 40600


# =========================
# GAZEBO SETTINGS
# =========================

WORLD_NAME = "sonar_seabed_world"
MODEL_NAME = "model_with_lidar"

# These must match your Gazebo DEM placement
TERRAIN_POSE_X = -20.875
TERRAIN_POSE_Y = -6.625

# Use forward mission origin for simple local alignment
SENSOR_Z_GAZEBO = 1.0

YAW_OFFSET_RAD = 0.0
PITCH_OFFSET_RAD = 0.0
ROLL_OFFSET_RAD = 0.0


class LidarReader(Node):
    def __init__(self):
        super().__init__("single_ping_lidar_reader")
        self.latest_scan = None
        self.create_subscription(LaserScan, "/lidar", self.cb, 10)

    def cb(self, msg):
        self.latest_scan = msg


def set_gazebo_pose(x, y, z, roll, pitch, yaw):
    quat = R.from_euler("xyz", [roll, pitch, yaw]).as_quat()
    qx, qy, qz, qw = quat

    req = f'''name: "{MODEL_NAME}"
position {{x: {x:.4f} y: {y:.4f} z: {z:.4f}}}
orientation {{x: {qx:.8f} y: {qy:.8f} z: {qz:.8f} w: {qw:.8f}}}'''

    cmd = [
        "gz", "service",
        "-s", f"/world/{WORLD_NAME}/set_pose",
        "--reqtype", "gz.msgs.Pose",
        "--reptype", "gz.msgs.Boolean",
        "--timeout", "1000",
        "--req", req
    ]

    subprocess.run(cmd)


def main():
    pos = pd.read_csv(POSE_CSV).sort_values("time_boot_ms")
    att = pd.read_csv(ATT_CSV).sort_values("time_boot_ms")
    sonar = pd.read_csv(REAL_SONAR_CSV)

    global PING_NUMBER

    if PING_NUMBER is None:
        counts = sonar.groupby("ping_number").size().sort_values(ascending=False)
        PING_NUMBER = int(counts.index[0])
        print("Auto-selected ping:", PING_NUMBER)
        print("Returns in this ping:", int(counts.iloc[0]))

    real_ping = sonar[sonar["ping_number"] == PING_NUMBER].copy()

    if len(real_ping) == 0:
        raise RuntimeError(f"No data found for ping {PING_NUMBER}")

    ping_time = real_ping["pwr_up_msec"].iloc[0]

    pose = pd.merge_asof(
        pos,
        att,
        on="time_boot_ms",
        direction="nearest"
    )

    pose_row = pd.merge_asof(
        pd.DataFrame({"time_boot_ms": [ping_time]}),
        pose,
        on="time_boot_ms",
        direction="nearest"
    ).iloc[0]

    X_MANUAL_OFFSET = -23.0
    Y_MANUAL_OFFSET = -1.0
    # Use forward mission local origin
    forward_x_min = pos["x_m"].min()
    forward_y_min = pos["y_m"].min()

    x_gz = pose_row["x_m"] - forward_x_min + TERRAIN_POSE_X + X_MANUAL_OFFSET
    y_gz = pose_row["y_m"] - forward_y_min + TERRAIN_POSE_Y + Y_MANUAL_OFFSET
    # Start with fixed sonar height above DEM
    z_gz = 0.0
    # z_gz = SENSOR_Z_GAZEBO

    # roll = pose_row["roll_rad"] + ROLL_OFFSET_RAD
    # pitch = pose_row["pitch_rad"] + PITCH_OFFSET_RAD
    # yaw = pose_row["yaw_rad"] + YAW_OFFSET_RAD

    # x_gz = 0.0
    # y_gz = 0.0
    # z_gz = 2.0

    # roll = 0.0
    # pitch = 0.0
    # yaw = 0.0

    PITCH_OFFSET_RAD = np.deg2rad(-30)
    # Use real attitude
    roll = pose_row["roll_rad"]
    pitch = pose_row["pitch_rad"] + PITCH_OFFSET_RAD
    yaw = pose_row["yaw_rad"]

    print("\nMoving virtual lidar to:")
    print("x_gz:", x_gz)
    print("y_gz:", y_gz)
    print("z_gz:", z_gz)
    print("roll pitch yaw:", roll, pitch, yaw)

    set_gazebo_pose(x_gz, y_gz, z_gz, roll, pitch, yaw)

    rclpy.init()
    node = LidarReader()

    scan = None
    for _ in range(50):
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.latest_scan is not None:
            scan = node.latest_scan
            break

    if scan is None:
        raise RuntimeError("No /lidar scan received. Is Gazebo running and playing?")

    rclpy.shutdown()

    # =========================
    # Virtual lidar scan
    # =========================

    virtual_ranges = np.array(scan.ranges)
    virtual_angles = scan.angle_min + np.arange(len(virtual_ranges)) * scan.angle_increment

    valid = np.isfinite(virtual_ranges)

    print("Total lidar rays:", len(scan.ranges))
    print("Valid finite rays:", valid.sum())
    print("Raw ranges:", scan.ranges)

    if valid.sum() == 0:
        print("No valid lidar hits.")
        print("Current pose:")
        print("x_gz =", x_gz)
        print("y_gz =", y_gz)
        print("z_gz =", z_gz)
        print("roll =", roll)
        print("pitch =", pitch)
        print("yaw =", yaw)
        return

    virtual_ranges = virtual_ranges[valid]
    virtual_angles = virtual_angles[valid]

    virtual_angle_deg = np.rad2deg(virtual_angles)

    virt_forward = virtual_ranges * np.cos(virtual_angles)
    virt_lateral = virtual_ranges * np.sin(virtual_angles)

    # =========================
    # Real sonar ping
    # =========================

    real_angles = np.deg2rad(real_ping["angle_deg"].to_numpy())
    real_ranges = real_ping["range_m"].to_numpy()

    real_forward = real_ranges * np.cos(real_angles)
    real_lateral = real_ranges * np.sin(real_angles)

    # =========================
    # Match nearest virtual angle for each real point
    # =========================

    match_rows = []

    for _, row in real_ping.iterrows():
        real_angle = row["angle_deg"]
        real_range = row["range_m"]

        idx = np.argmin(np.abs(virtual_angle_deg - real_angle))

        virt_angle = virtual_angle_deg[idx]
        virt_range = virtual_ranges[idx]

        angle_error = abs(virt_angle - real_angle)

        match_rows.append({
            "ping_number": PING_NUMBER,
            "real_angle_deg": real_angle,
            "virtual_angle_deg": virt_angle,
            "angle_error_deg": angle_error,
            "real_range_m": real_range,
            "virtual_range_m": virt_range,
            "error_m": real_range - virt_range
        })

    matched = pd.DataFrame(match_rows)
    matched.to_csv(f"single_ping_{PING_NUMBER}_comparison.csv", index=False)

    print("\nSaved:", f"single_ping_{PING_NUMBER}_comparison.csv")
    print(matched)
    print("\nPing MAE:", matched["error_m"].abs().mean())

    # =========================
    # Plot 1: sonar-frame spatial view
    # =========================

    plt.figure(figsize=(9, 7))

    plt.scatter(
        virt_lateral,
        virt_forward,
        s=25,
        label="Virtual LiDAR / DEM"
    )

    plt.scatter(
        real_lateral,
        real_forward,
        s=80,
        marker="x",
        label="Real forward-looking sonar"
    )

    plt.xlabel("Lateral distance from sensor (m)")
    plt.ylabel("Forward distance from sensor (m)")
    plt.title(f"Single-Ping Visual Comparison — Ping {PING_NUMBER}")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"single_ping_{PING_NUMBER}_spatial.png", dpi=300)
    plt.show()

    # =========================
    # Plot 2: range-angle view
    # =========================

    plt.figure(figsize=(9, 5))

    plt.plot(
        virtual_angle_deg,
        virtual_ranges,
        marker=".",
        label="Virtual LiDAR / DEM"
    )

    plt.scatter(
        real_ping["angle_deg"],
        real_ping["range_m"],
        s=80,
        marker="x",
        label="Real sonar returns"
    )

    plt.xlabel("Beam angle (deg)")
    plt.ylabel("Range (m)")
    plt.title(f"Range-Angle Comparison — Ping {PING_NUMBER}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"single_ping_{PING_NUMBER}_range_angle.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()