import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("2026-04-22-17-06.csv")

e = df["easting (UTM m)"].to_numpy()
n = df["northing (UTM m)"].to_numpy()
z = df["altitude (m)"].to_numpy()

# Remove NaN/inf
mask = np.isfinite(e) & np.isfinite(n) & np.isfinite(z)
e, n, z = e[mask], n[mask], z[mask]

# Local coordinates
x_local = e - e[0]
y_local = n - n[0]
coords = np.column_stack((x_local, y_local))

# Rotate to along-track/cross-track
coords_centered = coords - coords.mean(axis=0)
_, _, vh = np.linalg.svd(coords_centered, full_matrices=False)

x_along = coords_centered @ vh[0]
y_cross = coords_centered @ vh[1]
x_along = x_along - x_along.min()

# -----------------------------
# Outlier removal
# -----------------------------

# 1) Remove extreme depth values
z_low, z_high = np.percentile(z, [2, 98])
mask_depth = (z >= z_low) & (z <= z_high)

# 2) Remove far cross-track outliers
y_low, y_high = np.percentile(y_cross, [2, 98])
mask_cross = (y_cross >= y_low) & (y_cross <= y_high)

mask_clean = mask_depth & mask_cross

x_clean = x_along[mask_clean]
y_clean = y_cross[mask_clean]
z_clean = z[mask_clean]

# -----------------------------
# Representative curve through X-Z points
# -----------------------------

bin_size = 0.25  # meters; try 0.2, 0.5, 1.0
bins = np.arange(x_clean.min(), x_clean.max() + bin_size, bin_size)
bin_id = np.digitize(x_clean, bins)

x_curve = []
z_median = []
z_bottom = []

for b in np.unique(bin_id):
    m = bin_id == b

    if np.sum(m) > 10:
        x_curve.append(np.median(x_clean[m]))

        # middle trend
        z_median.append(np.median(z_clean[m]))

        # lower/deeper envelope, useful for obstacle/bottom boundary
        z_bottom.append(np.percentile(z_clean[m], 10))

x_curve = np.array(x_curve)
z_median = np.array(z_median)
z_bottom = np.array(z_bottom)

# Optional smoothing
from scipy.signal import medfilt

z_median_smooth = medfilt(z_median, kernel_size=9)
z_bottom_smooth = medfilt(z_bottom, kernel_size=9)

plt.figure()
plt.scatter(x_clean, z_clean, s=1, alpha=0.25, label="Cleaned sonar points")
plt.plot(x_curve, z_median_smooth, linewidth=2, label="Median curve")
#plt.plot(x_curve, z_bottom_smooth, linewidth=2, label="Bottom envelope")
plt.xlabel("Along-track distance X (m)")
plt.ylabel("Depth / Z (m)")
plt.title("X-Z Map with Fitted Depth Curve(Along track 90 forward)")
plt.grid(True)
plt.legend()
plt.show()

# -----------------------------
# Plot clearer bathymetry map
# -----------------------------
plt.figure()
plt.scatter(x_clean, y_clean, c=z_clean, s=1)
plt.colorbar(label="Depth / Z (m)")
plt.xlabel("Along-track distance (m)")
plt.ylabel("Across-track distance (m)")
plt.title("Cleaned Local Bathymetry Map")
plt.axis("equal")
plt.grid(True)
plt.show()

# -----------------------------
# Plot clearer X-Z profile
# -----------------------------
plt.figure()
plt.scatter(x_clean, z_clean, s=1)
plt.xlabel("Along-track distance X (m)")
plt.ylabel("Depth / Z (m)")
plt.title("Cleaned X-Z Bathymetry Profile")
plt.grid(True)
plt.show()

print("Points before cleaning:", len(z))
print("Points after cleaning:", len(z_clean))
print("Removed:", len(z) - len(z_clean))
print("Depth range after cleaning:", z_clean.min(), "to", z_clean.max())
