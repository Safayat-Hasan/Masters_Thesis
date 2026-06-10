import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

sonar = pd.read_csv("surveyor_atof.csv")
boat_pos = pd.read_csv("blueboat_local_position.csv")
boat_att = pd.read_csv("blueboat_attitude.csv")

boat_pos = boat_pos.rename(columns={
    "x_m": "boat_x_m",
    "y_m": "boat_y_m",
    "z_m": "boat_z_m"
})

# Reverse beam order inside each ping
sonar["beam_order"] = sonar.groupby("ping_number").cumcount()
sonar["beam_count"] = sonar.groupby("ping_number")["ping_number"].transform("count")
sonar["beam_order_reversed"] = sonar["beam_count"] - 1 - sonar["beam_order"]

# Time alignment
time_offset_ms = sonar["pwr_up_msec"].min() - boat_pos["time_boot_ms"].min()

sonar["t_sync"] = sonar["pwr_up_msec"]
boat_pos["t_sync"] = boat_pos["time_boot_ms"] + time_offset_ms
boat_att["t_sync"] = boat_att["time_boot_ms"] + time_offset_ms

sonar = sonar.sort_values("t_sync")
boat_pos = boat_pos.sort_values("t_sync")
boat_att = boat_att.sort_values("t_sync")

merged = pd.merge_asof(
    sonar,
    boat_pos[["t_sync", "boat_x_m", "boat_y_m", "boat_z_m"]],
    on="t_sync",
    direction="nearest",
    tolerance=100
)

merged = pd.merge_asof(
    merged,
    boat_att[["t_sync", "roll_rad", "pitch_rad", "yaw_rad"]],
    on="t_sync",
    direction="nearest",
    tolerance=100
)

merged = merged.dropna().copy()

# Sonar-frame points
# Important fix: flip y_m because SonarView beam order / swath side is opposite
p_sonar = np.vstack([
    np.zeros(len(merged)),
    -merged["y_m"].to_numpy(),
    merged["z_m"].to_numpy()
]).T

p_world = []

# Small mounting correction fitted from your SonarView CSV comparison
mount_rot = R.from_euler("xyz", [-2.72, 2.78, 0.0], degrees=True)

for k, row in enumerate(merged.itertuples(index=False)):
    boat_rot = R.from_euler(
        "xyz",
        [row.roll_rad, row.pitch_rad, row.yaw_rad],
        degrees=False
    )

    #p = boat_rot.apply(mount_rot.apply(p_sonar[k]))
    sonar_offset_boat = np.array([0.0, 0.16, 0.14])  # replace with measured sonar offset

    p_boat = mount_rot.apply(p_sonar[k]) + sonar_offset_boat
    p = boat_rot.apply(p_boat)
    
    p[0] += row.boat_x_m
    p[1] += row.boat_y_m

    p_world.append(p)

p_world = np.array(p_world)

merged["map_x_m"] = p_world[:, 0]
merged["map_y_m"] = p_world[:, 1]
merged["map_z_m"] = p_world[:, 2]

# SonarView-style altitude: negative below sonar
merged["altitude_like_sonarview_m"] = merged["map_z_m"]

merged.to_csv("Reconstructed_across_track_pointcloud.csv", index=False)

print("Saved Reconstructed_across_track_pointcloud.csv")
#print("Rows:", len(merged))

df = pd.read_csv("Reconstructed_across_track_pointcloud.csv")

# First and last ping numbers
first_ping = df["ping_number"].min()
last_ping = df["ping_number"].max()

# Rows belonging to first/last ping
start_ping = df[df["ping_number"] == first_ping]
end_ping = df[df["ping_number"] == last_ping]

# Swath-centre estimate using median
start_x = start_ping["map_y_m"].median()
start_y = start_ping["map_x_m"].median()

end_x = end_ping["map_y_m"].median()
end_y = end_ping["map_x_m"].median()

plt.figure(figsize=(8,6))
sc = plt.scatter(
    df["map_y_m"],
    df["map_x_m"],
    c=df["altitude_like_sonarview_m"],
    s=2
)

# Start marker
plt.scatter(
    start_x,
    start_y,
    color='red',
    s=120,
    marker='o',
    label='Start'
)

# End marker
plt.scatter(
    end_x,
    end_y,
    color='black',
    s=120,
    marker='X',
    label='End'
)

# Labels
plt.text(start_x, start_y, " START", color='red', fontsize=10)
plt.text(end_x, end_y, " END", color='black', fontsize=10)

plt.colorbar(sc, label="Altitude relative to sonar (m)")
plt.xlabel("Reconstructed Easting (m)")
plt.ylabel("Reconstructed Northing (m)")
plt.axis("equal")
plt.title("Reconstructed top-view point cloud")
plt.show()

# Project detections onto main mission direction using PCA
# Same style as your SonarView CSV side-view: no shifting to origin

XY = df[["map_x_m", "map_y_m"]].to_numpy()

# PCA only to estimate forward direction
origin = XY.mean(axis=0)
Xc = XY - origin
_, _, vh = np.linalg.svd(Xc, full_matrices=False)
forward = vh[0]

# Make sure forward axis follows mission time
first = df.nsmallest(200, "t_sync")[["map_x_m", "map_y_m"]].mean().to_numpy()
last = df.nlargest(200, "t_sync")[["map_x_m", "map_y_m"]].mean().to_numpy()

if np.dot(forward, last - first) < 0:
    forward = -forward

# Project using original local/map coordinates
df["forward_m"] = XY @ forward

# Do NOT shift to zero
# df["forward_m"] -= df["forward_m"].min()

df["z_m"] = df["altitude_like_sonarview_m"]

df = df[df["forward_m"] >= 60.0]

plt.figure(figsize=(10,5))
plt.scatter(df["forward_m"], df["z_m"], s=1, alpha=0.25)
plt.xlabel("Projected reconstructed detection position along forward motion (m)")
plt.ylabel("Altitude relative to sonar (m)")
plt.title("Side-view obstacle/seabed profile")
plt.grid(True)
plt.show()

bin_size = 0.25
df["bin"] = np.floor(df["forward_m"] / bin_size) * bin_size

profile = df.groupby("bin")["z_m"].agg(
    median_depth="median",
    deepest_point="min",
    count="count"
).reset_index()

profile = profile[profile["count"] >= 3]

plt.figure(figsize=(10,5))
plt.scatter(df["forward_m"], df["z_m"], s=1, alpha=0.08)
plt.plot(profile["bin"], profile["median_depth"], label="median profile")
#plt.plot(profile["bin"], profile["deepest_point"], label="deepest envelope")
plt.xlabel("Reconstructed sonar detection position along forward motion (m)")
plt.ylabel("Altitude relative to sonar (m)")
plt.title("Side-view of reconstructed obstacle/seabed profile with across-track sonar")
plt.grid(True)
plt.legend()
plt.show()

#sonar_offset_boat = np.array([0.0, 0.0, 0.0])  # replace with measured sonar offset

#p_boat = mount_rot.apply(p_sonar[k]) + sonar_offset_boat
#p = boat_rot.apply(p_boat)
