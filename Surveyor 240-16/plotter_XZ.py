import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("2026-04-22-15-40.csv")

# Use local detection positions
d = df[[
    "northing (local m)",
    "easting (local m)",
    "altitude (m)",
    "elapsed (s)"
]].dropna().copy()

XY = d[["northing (local m)", "easting (local m)"]].to_numpy()

# PCA: find main movement direction
origin = XY.mean(axis=0)
Xc = XY - origin
_, _, vh = np.linalg.svd(Xc, full_matrices=False)
forward = vh[0]

# Make sure forward axis follows time direction
first = d.nsmallest(200, "elapsed (s)")[["northing (local m)", "easting (local m)"]].mean().to_numpy()
last = d.nlargest(200, "elapsed (s)")[["northing (local m)", "easting (local m)"]].mean().to_numpy()

if np.dot(forward, last - first) < 0:
    forward = -forward

# Project detections onto forward direction
d["forward_m"] = XY @ forward
#d["forward_m"] -= d["forward_m"].min()

# Z/depth from sonar
d["z_m"] = d["altitude (m)"]

d = d[d["forward_m"] >= 60.0]

plt.figure(figsize=(10,5))
plt.scatter(d["forward_m"], d["z_m"], s=1, alpha=0.25)
plt.xlabel("Projected sonar detection position along forward motion (m)")
plt.ylabel("Altitude relative to sonar (m)")
plt.title("Side-view obstacle/seabed profile")
plt.grid(True)
plt.show()

bin_size = 0.25
d["bin"] = np.floor(d["forward_m"] / bin_size) * bin_size

profile = d.groupby("bin")["z_m"].agg(
    median_depth="median",
    deepest_point="min",
    count="count"
).reset_index()

profile = profile[profile["count"] >= 3]

plt.figure(figsize=(10,5))
plt.scatter(d["forward_m"], d["z_m"], s=1, alpha=0.08)
plt.plot(profile["bin"], profile["median_depth"], label="median profile")
#plt.plot(profile["bin"], profile["deepest_point"], label="deepest envelope")
plt.xlabel("Sonar detection position along forward motion (m)")
plt.ylabel("Altitude relative to sonar (m)")
plt.title("Side-view of obstacle/seabed profile with across-track sonar")
plt.grid(True)
plt.legend()
plt.show()
