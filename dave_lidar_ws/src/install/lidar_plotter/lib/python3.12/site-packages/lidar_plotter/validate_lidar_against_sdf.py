import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load saved lidar scan
df = pd.read_csv("lidar_scan_points.csv")

x = df["x_m"].values
z = df["z_m"].values

# ----------------------------
# Ground-truth values from SDF
# ----------------------------

# Lidar is at z = 0.5 m
# Ground plane box height = 0.1 m
# Ground top is z = 0.05 m in world
# Relative to lidar: 0.05 - 0.5 = -0.45 m
ground_z = -0.45

# Box center is around 4 m in front of lidar
# Box size is 1 m
# Near face is approximately at x = 3.5 m
box_front_x = 3.5

# Box vertical limits relative to lidar
# Box bottom = 0.0, top = 1.0, lidar height = 0.5
box_z_min = -0.5
box_z_max = 0.5

# ----------------------------
# Classify lidar points
# ----------------------------

# Points near the ground
ground_mask = z < -0.2

# Points near the obstacle face
box_mask = (x > 3.0) & (x < 4.2) & (z > box_z_min) & (z < box_z_max)

# ----------------------------
# Error calculation
# ----------------------------

ground_error = np.abs(z[ground_mask] - ground_z)
box_error = np.abs(x[box_mask] - box_front_x)

all_errors = np.concatenate([ground_error, box_error])

print("===== Quantitative Validation =====")
print(f"Ground points used: {len(ground_error)}")
print(f"Box points used: {len(box_error)}")
print(f"Total matched points: {len(all_errors)}")

if len(all_errors) > 0:
    print(f"MAE  = {np.mean(np.abs(all_errors)):.4f} m")
    print(f"RMSE = {np.sqrt(np.mean(all_errors**2)):.4f} m")
    print(f"Max error = {np.max(np.abs(all_errors)):.4f} m")
else:
    print("No valid points found for validation.")

# ----------------------------
# Plot validation
# ----------------------------

plt.figure()
plt.scatter(x, z, s=5, label="LiDAR points")

# Ground-truth ground line
plt.axhline(ground_z, linestyle="--", label="SDF ground truth")

# Ground-truth box front face
plt.plot(
    [box_front_x, box_front_x],
    [box_z_min, box_z_max],
    linestyle="--",
    label="SDF box front face"
)

plt.xlabel("Forward x (m)")
plt.ylabel("Vertical z (m)")
plt.title("LiDAR Points vs SDF Ground Truth")
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.show()