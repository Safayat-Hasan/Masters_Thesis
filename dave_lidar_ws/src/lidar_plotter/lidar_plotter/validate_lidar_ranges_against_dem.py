import numpy as np
import matplotlib.pyplot as plt

LIDAR_CSV = "lidar_scan_points.csv"
DEM_OBJ = "seabed_dem.obj"

# DEM pose from SDF
dem_pose = np.array([-20.875, -6.625, 0.0])

# LiDAR model pose from SDF
lidar_pos = np.array([0.0, 0.0, 2.0])

model_yaw = 3.14
sensor_pitch = 1.5708


def rot_y(p):
    c, s = np.cos(p), np.sin(p)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])


def rot_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])


def load_obj(path, offset):
    vertices = []
    faces = []

    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                vertices.append([
                    float(parts[1]) + offset[0],
                    float(parts[2]) + offset[1],
                    float(parts[3]) + offset[2]
                ])

            elif line.startswith("f "):
                parts = line.split()[1:]
                face = []
                for p in parts:
                    face.append(int(p.split("/")[0]) - 1)

                if len(face) == 3:
                    faces.append(face)
                elif len(face) == 4:
                    faces.append([face[0], face[1], face[2]])
                    faces.append([face[0], face[2], face[3]])

    return np.array(vertices), np.array(faces)


def ray_triangle_intersect(origin, direction, v0, v1, v2):
    eps = 1e-9

    edge1 = v1 - v0
    edge2 = v2 - v0

    h = np.cross(direction, edge2)
    a = np.dot(edge1, h)

    if -eps < a < eps:
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


def first_mesh_hit(origin, direction, vertices, faces):
    best_t = None

    for face in faces:
        v0, v1, v2 = vertices[face]

        t = ray_triangle_intersect(origin, direction, v0, v1, v2)

        if t is not None:
            if best_t is None or t < best_t:
                best_t = t

    return best_t


# Load data
scan = np.genfromtxt(LIDAR_CSV, delimiter=",", skip_header=1)
theta = scan[:, 1]
measured_range = scan[:, 3]

vertices, faces = load_obj(DEM_OBJ, dem_pose)

print("DEM mesh loaded")
print(f"Vertices: {len(vertices)}")
print(f"Triangles: {len(faces)}")

# Total sensor orientation
R = rot_z(model_yaw) @ rot_y(sensor_pitch)

expected_ranges = []
valid_measured = []

for th, r_meas in zip(theta, measured_range):

    # LiDAR horizontal scan direction in sensor local frame
    d_sensor = np.array([
        np.cos(th),
        np.sin(th),
        0.0
    ])

    # Transform ray direction to world
    d_world = R @ d_sensor
    d_world = d_world / np.linalg.norm(d_world)

    hit_range = first_mesh_hit(lidar_pos, d_world, vertices, faces)

    if hit_range is not None and np.isfinite(r_meas):
        expected_ranges.append(hit_range)
        valid_measured.append(r_meas)

expected_ranges = np.array(expected_ranges)
valid_measured = np.array(valid_measured)

error = valid_measured - expected_ranges


beam_id = np.arange(len(valid_measured))

# ==============================
# PLOT 1: Measured vs expected range
# ==============================

plt.figure()
plt.plot(beam_id, valid_measured, marker="o", markersize=3, label="Gazebo LiDAR measured range")
plt.plot(beam_id, expected_ranges, marker="x", markersize=3, label="DEM expected range")
plt.xlabel("Beam index")
plt.ylabel("Range (m)")
plt.title("Measured LiDAR Range vs DEM Ground-Truth Range")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ==============================
# PLOT 2: Range error per beam
# ==============================

plt.figure()
plt.plot(beam_id, error, marker="o", markersize=3)
plt.axhline(0, linestyle="--")
plt.xlabel("Beam index")
plt.ylabel("Range error (m)")
plt.title("Range Error Between LiDAR and DEM Ground Truth")
plt.grid(True)
plt.tight_layout()
plt.show()

# ==============================
# PLOT 3: Measured range vs expected range
# ==============================

plt.figure()
plt.scatter(expected_ranges, valid_measured, s=15)

min_r = min(expected_ranges.min(), valid_measured.min())
max_r = max(expected_ranges.max(), valid_measured.max())

plt.plot([min_r, max_r], [min_r, max_r], linestyle="--", label="Perfect agreement")
plt.xlabel("DEM expected range (m)")
plt.ylabel("Gazebo measured range (m)")
plt.title("Measured vs Expected Range Agreement")
plt.grid(True)
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.show()


print("\n===== Range-Based DEM Validation =====")
print(f"Matched beams: {len(error)}")

if len(error) > 0:
    print(f"MAE  = {np.mean(np.abs(error)):.4f} m")
    print(f"RMSE = {np.sqrt(np.mean(error**2)):.4f} m")
    print(f"Max error = {np.max(np.abs(error)):.4f} m")
else:
    print("No ray intersections found.")