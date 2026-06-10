import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator

# ==============================
# USER SETTINGS
# ==============================

LIDAR_CSV = "lidar_scan_points.csv"
DEM_OBJ = "seabed_dem.obj"

# DEM model pose from sonar_seabed.sdf
dem_pose_x = -20.875
dem_pose_y = -6.625
dem_pose_z = 0.0

# LiDAR model pose from SDF
lidar_x = 0.0
lidar_y = 0.0
lidar_z = 2.0

# Because model yaw is approximately pi,
# LiDAR forward direction points toward negative world x.
yaw_pi_forward_negative_x = True

# ==============================
# LOAD DEM OBJ VERTICES
# ==============================

vertices = []

with open(DEM_OBJ, "r") as f:
    for line in f:
        if line.startswith("v "):
            parts = line.strip().split()
            x, y, z = map(float, parts[1:4])
            vertices.append([x + dem_pose_x, y + dem_pose_y, z + dem_pose_z])

vertices = np.array(vertices)

dem_x = vertices[:, 0]
dem_y = vertices[:, 1]
dem_z = vertices[:, 2]

print("DEM loaded:")
print(f"  vertices: {len(vertices)}")
print(f"  x range: {dem_x.min():.3f} to {dem_x.max():.3f}")
print(f"  y range: {dem_y.min():.3f} to {dem_y.max():.3f}")
print(f"  z range: {dem_z.min():.3f} to {dem_z.max():.3f}")

# Create interpolation function z = DEM(x,y)
dem_interp = LinearNDInterpolator(
    list(zip(dem_x, dem_y)),
    dem_z
)

# ==============================
# LOAD LIDAR CSV
# ==============================

data = np.genfromtxt(
    LIDAR_CSV,
    delimiter=",",
    skip_header=1
)

local_forward = data[:, 4]  # x_m from CSV
local_z = data[:, 5]        # z_m from CSV


# ==============================
# CONVERT LIDAR LOCAL POINTS TO WORLD POINTS
# ==============================

if yaw_pi_forward_negative_x:
    world_x = lidar_x + local_forward
else:
    world_x = lidar_x + local_forward

world_y = np.full_like(world_x, lidar_y)
world_z = lidar_z + local_z

# ==============================
# COMPARE WITH DEM GROUND TRUTH
# ==============================

dem_z_at_lidar = dem_interp(world_x, world_y)

valid = np.isfinite(dem_z_at_lidar)

world_x_valid = world_x[valid]
world_y_valid = world_y[valid]
world_z_valid = world_z[valid]
dem_z_valid = dem_z_at_lidar[valid]

error = world_z_valid - dem_z_valid

print("\n===== DEM Validation =====")
print(f"Total LiDAR points: {len(world_x)}")
print(f"Valid points inside DEM: {len(error)}")

if len(error) > 0:
    print(f"MAE  = {np.mean(np.abs(error)):.4f} m")
    print(f"RMSE = {np.sqrt(np.mean(error**2)):.4f} m")
    print(f"Max error = {np.max(np.abs(error)):.4f} m")
else:
    print("No valid points fall inside the DEM area.")

# ==============================
# PLOT RESULT
# ==============================

plt.figure()
plt.scatter(world_x_valid, world_z_valid, s=8, label="LiDAR reconstructed points")
plt.scatter(world_x_valid, dem_z_valid, s=8, label="DEM ground truth")
plt.xlabel("World x (m)")
plt.ylabel("Height z (m)")
plt.title("LiDAR Points vs DEM Ground Truth")
plt.grid(True)
plt.legend()
plt.axis("equal")
plt.show()

plt.figure()
plt.scatter(world_x_valid, error, s=8)
plt.axhline(0, linestyle="--")
plt.xlabel("World x (m)")
plt.ylabel("Height error (m)")
plt.title("LiDAR Height Error Against DEM")
plt.grid(True)
plt.show()