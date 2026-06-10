import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# -----------------------------
# SETTINGS
# -----------------------------

POINT_MODE = "nearest"   # options: "all", "nearest", "deepest", "center"

SONAR_FILE = "surveyor_atof.csv"
POS_FILE   = "blueboat_local_position.csv"
ATT_FILE   = "blueboat_attitude.csv"

OUT_FILE = "forward_looking_pointcloud_world_with_mount.csv"

USE_ALL_PINGS = True
CENTER_PING = 39789
N_PINGS = 300

# Sonar mounting correction in boat frame
mount_roll_deg  = -2.72
mount_pitch_deg =  2.78
mount_yaw_deg   =  0.0

sonar_offset = np.array([
    0.0,    # x offset: forward
    0.16,   # y offset: starboard/lateral
    0.14    # z offset: down
])

# -----------------------------
# LOAD DATA
# -----------------------------
sonar = pd.read_csv(SONAR_FILE)
pos   = pd.read_csv(POS_FILE)
att   = pd.read_csv(ATT_FILE)

all_pings = np.sort(sonar["ping_number"].unique())

if USE_ALL_PINGS:
    selected_pings = all_pings
else:
    idx = np.where(all_pings == CENTER_PING)[0][0]
    selected_pings = all_pings[
        max(0, idx - N_PINGS // 2):
        min(len(all_pings), idx + N_PINGS // 2)
    ]

subset = sonar[sonar["ping_number"].isin(selected_pings)].copy()

# -----------------------------
# TIME ALIGNMENT
# -----------------------------
offset = sonar["pwr_up_msec"].min() - pos["time_boot_ms"].min()

print("Using pings:", len(selected_pings))
print("First ping:", selected_pings[0])
print("Last ping:", selected_pings[-1])
print("Offset:", offset)

# -----------------------------
# SONAR -> BOAT MOUNTING ROTATION
# -----------------------------
R_mount = R.from_euler(
    "xyz",
    [
        np.deg2rad(mount_roll_deg),
        np.deg2rad(mount_pitch_deg),
        np.deg2rad(mount_yaw_deg)
    ]
).as_matrix()

# -----------------------------
# RECONSTRUCT POINT CLOUD
# -----------------------------
world_points = []
boat_positions = []
rows = []

for ping_num in selected_pings:

    ping = subset[subset["ping_number"] == ping_num].copy()

    # --------------------------------------------------
    # Choose representative return per ping
    # --------------------------------------------------
    if POINT_MODE == "nearest":
        ping = ping.sort_values("range_m").head(1)

    elif POINT_MODE == "deepest":
        ping = ping.loc[[ping["y_m"].idxmin()]]

    elif POINT_MODE == "center":
        ping = ping.iloc[[len(ping) // 2]]

    elif POINT_MODE == "all":
        pass

    if len(ping) == 0:
        continue

    t_sonar = ping["pwr_up_msec"].iloc[0]
    t_boat = t_sonar - offset

    pos_i = pos.iloc[
        (pos["time_boot_ms"] - t_boat).abs().argmin()
    ]

    att_i = att.iloc[
        (att["time_boot_ms"] - t_boat).abs().argmin()
    ]

    boat_xyz = np.array([
        pos_i["x_m"],
        pos_i["y_m"],
        pos_i["z_m"]
    ])

    boat_positions.append(boat_xyz)

    # --------------------------------------------------
    # Surveyor raw forward-looking fan plane
    # --------------------------------------------------
    # This is the local sonar/instrument-frame assumption:
    # x_sonar = forward distance
    # y_sonar = lateral, forced 0 for thin top-view corridor
    # z_sonar = depth/down
    p_sonar = np.column_stack([
        -ping["z_m"].values,          # forward
        np.zeros(len(ping)),          # lateral
        -ping["y_m"].values           # down/depth
    ])

    # --------------------------------------------------
    # Sonar frame -> boat frame using mounting correction
    # --------------------------------------------------
    p_boat = (R_mount @ p_sonar.T).T + sonar_offset

    # --------------------------------------------------
    # Boat frame -> world frame
    # yaw + 180 verified from your yaw-case test
    # --------------------------------------------------
    R_world_boat = R.from_euler(
        "xyz",
        [
            att_i["roll_rad"],
            att_i["pitch_rad"],
            att_i["yaw_rad"] + np.pi
        ]
    ).as_matrix()

    p_world = (R_world_boat @ p_boat.T).T + boat_xyz

    world_points.append(p_world)

    for i in range(len(ping)):
        rows.append({
            "ping_number": ping_num,
            "pwr_up_msec": ping["pwr_up_msec"].iloc[i],

            "range_m": ping["range_m"].iloc[i],
            "angle_deg": ping["angle_deg"].iloc[i],
            "surveyor_y_m": ping["y_m"].iloc[i],
            "surveyor_z_m": ping["z_m"].iloc[i],

            "boat_x_m": boat_xyz[0],
            "boat_y_m": boat_xyz[1],
            "boat_z_m": boat_xyz[2],

            "x_sonar_m": p_sonar[i, 0],
            "y_sonar_m": p_sonar[i, 1],
            "z_sonar_m": p_sonar[i, 2],

            "x_boat_m": p_boat[i, 0],
            "y_boat_m": p_boat[i, 1],
            "z_boat_m": p_boat[i, 2],

            "map_x_m": p_world[i, 0],
            "map_y_m": p_world[i, 1],
            "map_z_m": p_world[i, 2],

            "roll_rad": att_i["roll_rad"],
            "pitch_rad": att_i["pitch_rad"],
            "yaw_rad_original": att_i["yaw_rad"],
            "yaw_rad_used": att_i["yaw_rad"] + np.pi
        })

world_points = np.vstack(world_points)
boat_positions = np.array(boat_positions)

df_out = pd.DataFrame(rows)

# -----------------------------
# FILTERS
# -----------------------------
df_out = df_out[
    (df_out["x_boat_m"] > 0.3) &
    (df_out["x_boat_m"] < 8.0) &
    (df_out["z_boat_m"] > 0.0) &
    (df_out["z_boat_m"] < 8.0)
].copy()

df_out["altitude_m"] = -df_out["z_boat_m"]

df_out.to_csv(OUT_FILE, index=False)

print("Saved:", OUT_FILE)
print("Saved points:", len(df_out))

# -----------------------------
# PLOT TOP VIEW
# -----------------------------
# plt.figure(figsize=(9, 8))

plt.plot(
    boat_positions[:, 1],
    boat_positions[:, 0],
    "r-",
    linewidth=1.5,
    label="Boat track"
)

sc = plt.scatter(
    df_out["map_y_m"],
    df_out["map_x_m"],
    c=df_out["altitude_m"],
    s=3,
    cmap="viridis",
    alpha=0.7,
    label="Sonar points"
)

plt.scatter(
    boat_positions[0, 1],
    boat_positions[0, 0],
    c="red",
    s=80,
    label="Start"
)

plt.scatter(
    boat_positions[-1, 1],
    boat_positions[-1, 0],
    c="black",
    marker="X",
    s=90,
    label="End"
)

plt.xlabel("Easting / local y (m)")
plt.ylabel("Northing / local x (m)")
plt.title("Forward-looking 90° sonar reconstructed point cloud")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.colorbar(sc, label="Altitude (m)")
plt.tight_layout()
plt.show()

