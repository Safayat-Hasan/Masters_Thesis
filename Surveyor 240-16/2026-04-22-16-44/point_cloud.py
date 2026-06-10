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
theta = np.deg2rad(merged["angle_deg"].to_numpy())

ranges = merged["range_m"].to_numpy()

x_sonar = ranges * np.cos(theta)
y_sonar = np.zeros(len(merged))
z_sonar = ranges * np.sin(theta)

p_sonar = np.vstack([
    -merged["z_m"].to_numpy(),
    np.zeros(len(merged)),
    merged["y_m"].to_numpy()
]).T

p_world = []
p_boat_all = []

mount_rot = R.from_euler("xyz", [-2.72, 2.78, 0.0], degrees=True)

for k, row in enumerate(merged.itertuples(index=False)):

    boat_rot = R.from_euler(
        "xyz",
        [row.roll_rad, row.pitch_rad, row.yaw_rad],
        degrees=False
    )

    sonar_offset_boat = np.array([0.0, 0.16, 0.14])

    p_boat = mount_rot.apply(p_sonar[k]) + sonar_offset_boat

    # SAVE BOAT-FRAME POINT
    p_boat_all.append(p_boat.copy())

    p = boat_rot.apply(p_boat)

    p[0] += row.boat_x_m
    p[1] += row.boat_y_m

    p_world.append(p)

p_world = np.array(p_world)
p_boat_all = np.array(p_boat_all)

merged["map_x_m"] = p_world[:, 0]
merged["map_y_m"] = p_world[:, 1]
merged["map_z_m"] = p_world[:, 2]

# save boat-frame points too
merged["x_boat_m"] = p_boat_all[:, 0]
merged["y_boat_m"] = p_boat_all[:, 1]
merged["z_boat_m"] = p_boat_all[:, 2]

# SonarView-style altitude: negative below sonar
merged["altitude_like_sonarview_m"] = merged["map_z_m"]

# ── DEDUPLICATE ───────────────────────────────────────────────────────────
# Forward-looking sonar sees the same world point repeatedly as boat approaches.
# Bin by world-space (map_x, map_y) position, keep closest-range observation.
merged["range_m"] = np.sqrt(
    (merged["map_x_m"] - merged["boat_x_m"])**2 +
    (merged["map_y_m"] - merged["boat_y_m"])**2
)

dedup_bin = 0.25  # metres — tune to sonar resolution
merged["nx_bin"] = np.floor(merged["map_x_m"] / dedup_bin) * dedup_bin
merged["ny_bin"] = np.floor(merged["map_y_m"] / dedup_bin) * dedup_bin

merged = (
    merged.sort_values("range_m")
          .groupby(["nx_bin", "ny_bin"], as_index=False)
          .first()
)
print(f"After deduplication: {len(merged)} points")
# ─────────────────────────────────────────────────────────────────────────

merged.to_csv("Reconstructed_forward_pointcloud.csv", index=False)

print("Saved Reconstructed_forward_pointcloud.csv")
#print("Rows:", len(merged))

df = pd.read_csv("Reconstructed_forward_pointcloud.csv")

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

# ---------- CLEAN FORWARD-LOOKING PROFILE ----------

df = pd.read_csv("Reconstructed_forward_pointcloud.csv")

# Estimate boat path direction using boat positions, not detection positions
boat_xy = df[["boat_x_m", "boat_y_m"]].drop_duplicates().to_numpy()

origin = boat_xy.mean(axis=0)
Xc = boat_xy - origin

_, _, vh = np.linalg.svd(Xc, full_matrices=False)
track_axis = vh[0]

# Make direction follow time
first = df.nsmallest(200, "t_sync")[["boat_x_m", "boat_y_m"]].mean().to_numpy()
last = df.nlargest(200, "t_sync")[["boat_x_m", "boat_y_m"]].mean().to_numpy()

if np.dot(track_axis, last - first) < 0:
    track_axis = -track_axis

# Boat position along mission track
boat_xy_all = df[["boat_x_m", "boat_y_m"]].to_numpy()
df["boat_s_m"] = boat_xy_all @ track_axis

# Combined obstacle position:
# where the boat was + how far ahead the sonar saw
df["obstacle_s_m"] = df["boat_s_m"] + df["x_boat_m"]

# Optional filtering
d = df[
    (df["x_boat_m"] > 0.2) &
    (df["x_boat_m"] < 8.0)
].copy()

plt.figure(figsize=(12,5))
plt.scatter(
    d["obstacle_s_m"],
    d["z_boat_m"],
    s=2,
    alpha=0.2
)
plt.xlabel("Obstacle position along mission track (m)")
plt.ylabel("Vertical distance in boat/sonar frame (m)")
plt.title("Combined forward-looking side-view obstacle map")
plt.grid(True)
plt.show()

bin_size = 0.25

d["bin"] = np.floor(d["obstacle_s_m"] / bin_size) * bin_size

profile = d.groupby("bin")["z_boat_m"].agg(
    median_depth="median",
    deepest="min",
    shallowest="max",
    count="count"
).reset_index()

profile = profile[profile["count"] >= 3]

profile["median_smooth"] = profile["median_depth"].rolling(
    window=7, center=True, min_periods=1
).median()

plt.figure(figsize=(12,5))
plt.scatter(d["obstacle_s_m"], d["z_boat_m"], s=1, alpha=0.05)
plt.plot(profile["bin"], profile["median_smooth"], linewidth=2, label="median profile")
plt.xlabel("Obstacle position along mission track (m)")
plt.ylabel("Vertical distance in boat/sonar frame (m)")
plt.title("Clean combined forward-looking side-view obstacle map")
plt.grid(True)
plt.legend()
plt.show()

#sonar_offset_boat = np.array([0.0, 0.0, 0.0])  # replace with measured sonar offset

#p_boat = mount_rot.apply(p_sonar[k]) + sonar_offset_boat
#p = boat_rot.apply(p_boat)
