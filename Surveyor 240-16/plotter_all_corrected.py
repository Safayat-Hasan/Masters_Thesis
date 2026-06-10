import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("2026-04-22-16-15.csv")

xcol = "easting (local m)"
ycol = "northing (local m)"
zcol = "altitude (m)"
pcol = "ping number"

# 1. Estimate trajectory from ping centroids
traj = (
    df.groupby(pcol)[[xcol, ycol]]
    .mean()
    .reset_index()
    .sort_values(pcol)
)

# Smooth trajectory a little
traj["x_s"] = traj[xcol].rolling(21, center=True, min_periods=1).mean()
traj["y_s"] = traj[ycol].rolling(21, center=True, min_periods=1).mean()

# 2. Estimate heading from trajectory
dx = np.gradient(traj["x_s"])
dy = np.gradient(traj["y_s"])

heading = np.arctan2(dy, dx)

traj["tx"] = np.cos(heading)   # along-track unit vector x
traj["ty"] = np.sin(heading)   # along-track unit vector y

traj["nx"] = -np.sin(heading)  # left-normal unit vector x
traj["ny"] =  np.cos(heading)  # left-normal unit vector y

# 3. Merge trajectory info back to every sonar point
df = df.merge(
    traj[[pcol, "x_s", "y_s", "tx", "ty", "nx", "ny"]],
    on=pcol,
    how="left"
)

# 4. Offset of each point from ping centroid
rx = df[xcol] - df["x_s"]
ry = df[ycol] - df["y_s"]

# Components in path frame
along = rx * df["tx"] + ry * df["ty"]
cross = rx * df["nx"] + ry * df["ny"]

# ---------------------------------------------------
# TRUE forward-looking correction
# ---------------------------------------------------

# Forward-looking sonar should have VERY SMALL
# cross-track spread.

# Keep along-track information
along_corr = along

# Compress cross-track heavily
compression = 0.03   # try 0.01 to 0.1

cross_corr = compression * cross

# Reconstruct corrected coordinates
df["x_corrected"] = (
    df["x_s"]
    + along_corr * df["tx"]
    + cross_corr * df["nx"]
)

df["y_corrected"] = (
    df["y_s"]
    + along_corr * df["ty"]
    + cross_corr * df["ny"]
)

# 6. Plot corrected forward-looking bathymetry
plt.figure(figsize=(7, 6))
sc = plt.scatter(
    df["x_corrected"],
    df["y_corrected"],
    c=df[zcol],
    s=3,
    cmap="viridis"
)
plt.colorbar(sc, label="Depth / altitude (m)")
plt.axis("equal")
plt.xlabel("Easting local corrected (m)")
plt.ylabel("Northing local corrected (m)")
plt.title("Corrected Forward-Looking Sonar Map")
plt.show()
