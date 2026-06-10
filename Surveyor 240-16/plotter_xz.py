import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("2026-04-22-16-15.csv")

d = df[[
    "northing (local m)",
    "easting (local m)",
    "altitude (m)",
    "elapsed (s)"
]].dropna().copy()

XY = d[["northing (local m)", "easting (local m)"]].to_numpy()

# PCA: find main movement direction (same as before)
origin = XY.mean(axis=0)
Xc = XY - origin
_, _, vh = np.linalg.svd(Xc, full_matrices=False)
forward = vh[0]
lateral = vh[1]  # <-- NEW: cross-track axis (perpendicular to forward)

# Ensure forward follows time direction
first = d.nsmallest(200, "elapsed (s)")[["northing (local m)", "easting (local m)"]].mean().to_numpy()
last  = d.nlargest(200,  "elapsed (s)")[["northing (local m)", "easting (local m)"]].mean().to_numpy()
if np.dot(forward, last - first) < 0:
    forward = -forward
    lateral = -lateral  # keep consistent handedness

# Project onto both axes
d["forward_m"] = (XY - origin) @ forward   # along-track position of detection
d["lateral_m"] = (XY - origin) @ lateral   # cross-track position of detection
d["z_m"]       = d["altitude (m)"]         # altitude/depth from sonar

# ── FORWARD-LOOKING SPECIFIC: deduplicate repeat observations ──────────────
#
# For a forward-looker, the sonar vehicle position at time of ping AND
# the detected point position are both in the data. The range to the
# target is approximately the forward_m of the detection minus the
# forward_m of the boat at that ping time.
#
# Simple approach: bin detections by their (lateral, z) world-space location
# and keep only the detection seen at minimum range (closest approach).
# "Range" here = distance from boat position to detection position.
# Since we don't have boat position separately, we approximate:
# as the boat approaches, forward_m of detection stays ~constant but
# the boat's own forward position increases — so LATER pings of the same
# object have smaller range. We keep the LAST (latest elapsed time) ping
# per spatial bin, which is the closest-approach observation.

bin_size = 0.5  # metres — tune to your sonar resolution

d["lat_bin"] = np.floor(d["lateral_m"] / bin_size) * bin_size
d["z_bin"]   = np.floor(d["z_m"]       / bin_size) * bin_size

# Keep closest-approach ping per (lateral, depth) cell
# = the one with the largest elapsed time (boat was nearest)
d_dedup = (
    d.sort_values("elapsed (s)")
     .groupby(["lat_bin", "z_bin"], as_index=False)
     .last()   # last ping = closest approach for forward-looker
)

# ── PLOT ──────────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 5))
plt.scatter(d_dedup["lateral_m"], d_dedup["z_m"], s=2, alpha=0.4)
plt.xlabel("Cross-track position (m)  [lateral spread of detections]")
plt.ylabel("Altitude / range (m)")
plt.title("Forward-looking sonar — side view (deduplicated)")
plt.grid(True)
plt.show()

# ── PROFILE with median + envelope ────────────────────────────────────────
bin_size_profile = 0.25
d_dedup["bin"] = np.floor(d_dedup["lateral_m"] / bin_size_profile) * bin_size_profile

profile = d_dedup.groupby("bin")["z_m"].agg(
    median_depth="median",
    deepest_point="min",
    count="count"
).reset_index()
profile = profile[profile["count"] >= 3]

plt.figure(figsize=(10, 5))
plt.scatter(d_dedup["lateral_m"], d_dedup["z_m"], s=1, alpha=0.15)
plt.plot(profile["bin"], profile["median_depth"],  label="median profile")
plt.plot(profile["bin"], profile["deepest_point"], label="deepest envelope")
plt.xlabel("Cross-track position (m)")
plt.ylabel("Altitude / range (m)")
plt.title("Forward-looking sonar — clean side-view profile")
plt.grid(True)
plt.legend()
plt.show()
