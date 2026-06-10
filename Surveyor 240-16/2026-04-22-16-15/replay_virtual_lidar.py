import subprocess
import time
import numpy as np
import pandas as pd

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from scipy.spatial.transform import Rotation as R


# =========================
# FILE PATHS
# =========================

POINTCLOUD_CSV = "Reconstructed_across_track_pointcloud.csv"
POSE_CSV = "blueboat_local_position.csv"
ATT_CSV = "blueboat_attitude.csv"
REAL_SONAR_CSV = "surveyor_atof.csv"

OUT_VIRTUAL = "virtual_lidar_ranges.csv"
OUT_COMPARE = "real_vs_virtual_ranges.csv"


# =========================
# GAZEBO SETTINGS
# =========================

WORLD_NAME = "sonar_seabed_world"
MODEL_NAME = "model_with_lidar"

# DEM mesh was shifted to origin when OBJ was created.
# This pose must match your seabed_dem pose in SDF.
TERRAIN_POSE_X = -20.875
TERRAIN_POSE_Y = -6.625

# Start simple: put virtual sonar 2 m above Gazebo origin.
# Later tune this based on real sonar height.
SENSOR_Z_GAZEBO = 2.0

# Optional yaw correction if lidar does not point forward correctly
YAW_OFFSET_RAD = 0.0


class LidarReader(Node):
    def __init__(self):
        super().__init__("lidar_reader")
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

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    # =========================
    # LOAD DATA
    # =========================

    pc = pd.read_csv(POINTCLOUD_CSV)
    pos = pd.read_csv(POSE_CSV).sort_values("time_boot_ms")
    att = pd.read_csv(ATT_CSV).sort_values("time_boot_ms")
    sonar = pd.read_csv(REAL_SONAR_CSV)

    forward_x_min = pos["x_m"].min()
    forward_y_min = pos["y_m"].min()

    # merge boat position + attitude
    pose = pd.merge_asof(
        pos.sort_values("time_boot_ms"),
        att.sort_values("time_boot_ms"),
        on="time_boot_ms",
        direction="nearest"
    )

    # one row per sonar ping time
    ping_times = (
        sonar[["pwr_up_msec", "ping_number"]]
        .drop_duplicates()
        .rename(columns={"pwr_up_msec": "time_boot_ms"})
        .sort_values("time_boot_ms")
    )

    ping_pose = pd.merge_asof(
        ping_times,
        pose.sort_values("time_boot_ms"),
        on="time_boot_ms",
        direction="nearest"
    )

    print("Number of pings to replay:", len(ping_pose))

    # =========================
    # ROS LIDAR SUBSCRIBER
    # =========================

    rclpy.init()
    node = LidarReader()

    virtual_rows = []

    for i, row in ping_pose.head(100).iterrows():
        # Original map frame -> Gazebo DEM frame
        x_gz = row["x_m"] - forward_x_min + TERRAIN_POSE_X
        y_gz = row["y_m"] - forward_y_min + TERRAIN_POSE_Y
        z_gz = SENSOR_Z_GAZEBO

        roll = row["roll_rad"]
        pitch = row["pitch_rad"]
        yaw = row["yaw_rad"] + YAW_OFFSET_RAD

        set_gazebo_pose(x_gz, y_gz, z_gz, roll, pitch, yaw)
        time.sleep(0.05)

        # wait for updated scan
        scan = None
        for _ in range(20):
            rclpy.spin_once(node, timeout_sec=0.05)
            if node.latest_scan is not None:
                scan = node.latest_scan
                break

        if scan is None:
            print("No scan received")
            continue

        ranges = np.array(scan.ranges)
        angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment

        valid = np.isfinite(ranges)

        for beam_idx, (ang, rng, ok) in enumerate(zip(angles, ranges, valid)):
            if ok:
                virtual_rows.append({
                    "ping_number": row["ping_number"],
                    "time_boot_ms": row["time_boot_ms"],
                    "beam_idx": beam_idx,
                    "virtual_angle_rad": ang,
                    "virtual_angle_deg": np.degrees(ang),
                    "virtual_range_m": rng,
                    "x_gazebo": x_gz,
                    "y_gazebo": y_gz,
                    "z_gazebo": z_gz,
                    "yaw_rad": yaw
                })

        if i % 20 == 0:
            print(f"Processed {i}/{len(ping_pose)}")

    virtual = pd.DataFrame(virtual_rows)
    virtual.to_csv(OUT_VIRTUAL, index=False)

    # =========================
    # COMPARE REAL VS VIRTUAL
    # =========================

    compare_rows = []

    for _, real in sonar.iterrows():
        ping = real["ping_number"]
        real_angle = real["angle_deg"]
        real_range = real["range_m"]

        subset = virtual[virtual["ping_number"] == ping]

        if len(subset) == 0:
            continue

        idx = (subset["virtual_angle_deg"] - real_angle).abs().idxmin()
        virt = subset.loc[idx]

        angle_error = abs(
            virt["virtual_angle_deg"] - real_angle
        )

        # Skip if matched virtual ray is too far away
        if angle_error > 1.0:
            continue

        compare_rows.append({
            "ping_number": ping,
            "real_angle_deg": real_angle,
            "real_range_m": real_range,
            "virtual_angle_deg": virt["virtual_angle_deg"],
            "angle_error_deg": angle_error,
            "virtual_range_m": virt["virtual_range_m"],
            "error_m": real_range - virt["virtual_range_m"]
        })

    compare = pd.DataFrame(compare_rows)
    compare.to_csv(OUT_COMPARE, index=False)

    print("Saved:", OUT_VIRTUAL)
    print("Saved:", OUT_COMPARE)

    if len(compare) > 0:
        print("MAE:", compare["error_m"].abs().mean())
        print("RMSE:", np.sqrt((compare["error_m"] ** 2).mean()))

    

    df = pd.read_csv("real_vs_virtual_ranges.csv")

    print("Rows:", len(df))
    print("Max angle error:", df["angle_error_deg"].max())
    print("Mean angle error:", df["angle_error_deg"].mean())
    print("MAE:", df["error_m"].abs().mean())
    print("RMSE:", (df["error_m"]**2).mean()**0.5)

    rclpy.shutdown()


if __name__ == "__main__":
    main()