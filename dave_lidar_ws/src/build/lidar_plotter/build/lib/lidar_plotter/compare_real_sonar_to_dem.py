import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# ==============================
# FILES
# ==============================

SONAR_CSV = "surveyor_atof.csv"
POS_CSV = "blueboat_local_position.csv"
ATT_CSV = "blueboat_attitude.csv"
DEM_OBJ = "seabed_dem.obj"

# Choose one ping first
PING_NUMBER = 39789

# ==============================
# DEM POSE IN GAZEBO WORLD
# ==============================

dem_pose = np.array([-20.875, -6.625, 0.0])

# IMPORTANT:
# Your BlueBoat local position is around x=37, y=19.
# Your DEM world is around x=-20.875..20.875, y=-6.625..6.625.
# So we must center BlueBoat local frame into DEM frame.
#
# Start with these as a first approximation:
DEM_CENTER_X_LOCAL = 37.55
DEM_CENTER_Y_LOCAL = 19.45

# ==============================
# SONAR MOUNT ON BLUEBOAT
# ==============================

sonar_offset_boat = np.array([0.0, 0.16, 0.14])

mount_roll_deg = -2.72
mount_pitch_deg = 2.78
mount_yaw_deg = 0.0

R_mount = R.from_euler(
    "xyz",
    [mount_roll_deg, mount_pitch_deg, mount_yaw_deg],
    degrees=True
).as_matrix()

# ==============================
# RAY / MESH FUNCTIONS
# ==============================

def load_obj(path, offset):
    
    vertices = []
    faces = []

    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                p = line.strip().split()
                vertices.append([
                    float(p[1]) + offset[0],
                    float(p[2]) + offset[1],
                    float(p[3]) + offset[2],
                ])

            elif line.startswith("f "):
                parts = line.strip().split()[1:]
                idx = [int(x.split("/")[0]) - 1 for x in parts]

                if len(idx) == 3:
                    faces.append(idx)
                elif len(idx) == 4:
                    faces.append([idx[0], idx[1], idx[2]])
                    faces.append([idx[0], idx[2], idx[3]])

    return np.array(vertices), np.array(faces)


def ray_triangle_intersect(origin, direction, v0, v1, v2):
    eps = 1e-9

    edge1 = v1 - v0
    edge2 = v2 - v0

    h = np.cross(direction, edge2)
    a = np.dot(edge1, h)

    if abs(a) < eps:
        return None

    f = 1.0 / a
    s = origin - v0
    u = f * np.dot(s, h)

    if u < 0.0 or u > 1.0:
        return None

    q = np.cross(s, edge1)
    v = f * np.dot(direction, q)

    if v < 0.0 or u + v > 1.0:
        return None

    t = f * np.dot(edge2, q)

    if t > eps:
        return t

    return None




def first_mesh_hit_fast(origin, direction, tri_v0, tri_v1, tri_v2):
    eps = 1e-9

    edge1 = tri_v1 - tri_v0
    edge2 = tri_v2 - tri_v0

    h = np.cross(direction, edge2)
    a = np.einsum("ij,ij->i", edge1, h)

    valid = np.abs(a) > eps
    if not np.any(valid):
        return None

    f = np.zeros_like(a)
    f[valid] = 1.0 / a[valid]

    s = origin - tri_v0
    u = f * np.einsum("ij,ij->i", s, h)

    valid &= (u >= 0.0) & (u <= 1.0)

    q = np.cross(s, edge1)
    v = f * np.einsum("j,ij->i", direction, q)

    valid &= (v >= 0.0) & ((u + v) <= 1.0)

    t = f * np.einsum("ij,ij->i", edge2, q)

    valid &= t > eps

    if not np.any(valid):
        return None

    return np.min(t[valid])


# ==============================
# LOAD DATA
# ==============================

sonar = pd.read_csv(SONAR_CSV)
pos = pd.read_csv(POS_CSV)
att = pd.read_csv(ATT_CSV)

vertices, faces = load_obj(DEM_OBJ, dem_pose)

tri_v0 = vertices[faces[:, 0]]
tri_v1 = vertices[faces[:, 1]]
tri_v2 = vertices[faces[:, 2]]

print("DEM mesh loaded")
print(f"Vertices: {len(vertices)}")
print(f"Triangles: {len(faces)}")

# ==============================
# SELECT ONE PING
# ==============================

ping_df = sonar[sonar["ping_number"] == PING_NUMBER].copy()

if len(ping_df) == 0:
    raise RuntimeError(f"No sonar data found for ping {PING_NUMBER}")

ping_time = ping_df["pwr_up_msec"].iloc[0]

# Time offset between sonar and BlueBoat logs
time_offset = sonar["pwr_up_msec"].min() - pos["time_boot_ms"].min()
boat_time = ping_time - time_offset

print("\nSelected ping:")
print(f"Ping number: {PING_NUMBER}")
print(f"Sonar pwr_up_msec: {ping_time}")
print(f"Estimated boat time_boot_ms: {boat_time:.1f}")

# Nearest BlueBoat position and attitude
pos_idx = (pos["time_boot_ms"] - boat_time).abs().idxmin()
att_idx = (att["time_boot_ms"] - boat_time).abs().idxmin()

pos_row = pos.loc[pos_idx]
att_row = att.loc[att_idx]

boat_local = np.array([
    pos_row["x_m"],
    pos_row["y_m"],
    pos_row["z_m"]
])

roll = att_row["roll_rad"]
pitch = att_row["pitch_rad"]
yaw = att_row["yaw_rad"]

print("\nMatched BlueBoat pose:")
print(f"local position x,y,z = {boat_local}")
print(f"roll,pitch,yaw rad = {roll:.3f}, {pitch:.3f}, {yaw:.3f}")

# Convert BlueBoat local x,y into DEM/Gazebo frame
boat_world_xy = np.array([
    boat_local[0] - DEM_CENTER_X_LOCAL,
    boat_local[1] - DEM_CENTER_Y_LOCAL
])

# For first comparison, use fixed sensor height above DEM
# Later you can improve this using true altitude/MSL alignment.
R_boat = R.from_euler("xyz", [roll, pitch, yaw]).as_matrix()

def compute_for_sensor_z(sensor_z, angle_sign, boat_world_xy):

    boat_world = np.array([
        boat_world_xy[0],
        boat_world_xy[1],
        sensor_z
    ])

    sonar_origin_world = boat_world + R_boat @ sonar_offset_boat

    rows = []

    for _, row in ping_df.iterrows():

        angle = angle_sign * row["angle_rad"]
        real_range = row["range_m"]

        d_sonar = np.array([
            np.cos(angle),
            0.0,
            np.sin(angle)
        ])

        d_world = R_boat @ R_mount @ d_sonar
        d_world = d_world / np.linalg.norm(d_world)

        expected_range = first_mesh_hit_fast(
            sonar_origin_world,
            d_world,
            tri_v0,
            tri_v1,
            tri_v2
        )

        if expected_range is None:
            continue

        rows.append({
            "beam_id": int(row["point_index"]),
            "angle_deg": row["angle_deg"],
            "real_sonar_range_m": real_range,
            "dem_expected_range_m": expected_range,
            "error_real_minus_dem_m": real_range - expected_range
        })

    return pd.DataFrame(rows), sonar_origin_world

best = None

x_offsets = np.linspace(36.5, 38.5, 41)
y_offsets = np.linspace(18.5, 20.5, 41)
z_values = np.linspace(-0.8, 0.3, 45)

for angle_sign in [1.0, -1.0]:
    print("Testing angle sign:", angle_sign)

    for cx in x_offsets:
        for cy in y_offsets:
            boat_world_xy = np.array([
                boat_local[0] - cx,
                boat_local[1] - cy
            ])

            for sensor_z in z_values:

                test_df, test_origin = compute_for_sensor_z(
                    sensor_z,
                    angle_sign,
                    boat_world_xy
                )

                if len(test_df) < 5:
                    continue

                mae = test_df["error_real_minus_dem_m"].abs().mean()

                if best is None or mae < best["mae"]:
                    best = {
                        "mae": mae,
                        "cx": cx,
                        "cy": cy,
                        "sensor_z": sensor_z,
                        "angle_sign": angle_sign,
                        "df": test_df,
                        "origin": test_origin
                    }

if best is None:
    raise RuntimeError("No valid DEM intersections found for any tested sensor height.")

out = best["df"]
sonar_origin_world = best["origin"]

print("\n===== Best Real Sonar vs DEM Alignment =====")
print(f"Best DEM_CENTER_X_LOCAL: {best['cx']:.3f}")
print(f"Best DEM_CENTER_Y_LOCAL: {best['cy']:.3f}")
print(f"Best sensor_z: {best['sensor_z']:.3f}")
print(f"Best angle_sign: {best['angle_sign']}")
print(f"Sonar origin: {best['origin']}")
print(f"Matched beams: {len(out)}")
print(f"MAE  = {out['error_real_minus_dem_m'].abs().mean():.4f} m")
print(f"RMSE = {np.sqrt((out['error_real_minus_dem_m']**2).mean()):.4f} m")
print(f"Max error = {out['error_real_minus_dem_m'].abs().max():.4f} m")

out.to_csv("real_sonar_vs_dem_one_ping.csv", index=False)

print("\nSonar origin used in DEM/Gazebo frame:")
print(sonar_origin_world)


# ==============================
# PLOTS
# ==============================

plt.figure()
plt.plot(out["angle_deg"], out["real_sonar_range_m"], marker="o", label="Real sonar range")
plt.plot(out["angle_deg"], out["dem_expected_range_m"], marker="x", label="DEM expected range")
plt.xlabel("Beam angle (deg)")
plt.ylabel("Range (m)")
plt.title(f"Real Sonar vs DEM Expected Range, Ping {PING_NUMBER}")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("real_sonar_vs_dem_range.png", dpi=300)
plt.show()

plt.figure()
plt.plot(out["angle_deg"], out["error_real_minus_dem_m"], marker="o")
plt.axhline(0, linestyle="--")
plt.xlabel("Beam angle (deg)")
plt.ylabel("Range error (m)")
plt.title(f"Real Sonar - DEM Expected Error, Ping {PING_NUMBER}")
plt.grid(True)
plt.tight_layout()
plt.savefig("real_sonar_vs_dem_error.png", dpi=300)
plt.show()