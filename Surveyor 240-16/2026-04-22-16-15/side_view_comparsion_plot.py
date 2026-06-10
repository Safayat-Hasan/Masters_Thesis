import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# FILES
# -----------------------------
FL_FILE = "forward_looking_pointcloud_world_with_mount.csv"
AT_FILE = "Reconstructed_across_track_pointcloud.csv"

# Use same physical coordinate range as top-view plot
X_MIN = 60.0
X_MAX = 85.0

spatial_bin = 0.25
profile_bin = 0.25
depth_match_tol = 0.35


# -----------------------------
# HELPERS
# -----------------------------
def get_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Missing columns. Tried: {candidates}")


def choose_depth_column(df):
    if "altitude_m" in df.columns:
        return "altitude_m"
    if "altitude_like_sonarview_m" in df.columns:
        return "altitude_like_sonarview_m"
    if "z_boat_m" in df.columns:
        df["altitude_m"] = -df["z_boat_m"]
        return "altitude_m"
    if "z_m" in df.columns:
        return "z_m"
    if "map_z_m" in df.columns:
        return "map_z_m"
    raise KeyError("No depth column found")


# -----------------------------
# LOAD
# -----------------------------
fl_raw = pd.read_csv(FL_FILE)
at_raw = pd.read_csv(AT_FILE)

fl_x = get_col(fl_raw, ["map_x_m", "x_m", "northing_m", "northing (local m)"])
fl_y = get_col(fl_raw, ["map_y_m", "y_m", "easting_m", "easting (local m)"])
fl_z = choose_depth_column(fl_raw)

at_x = get_col(at_raw, ["map_x_m", "x_m", "northing_m", "northing (local m)"])
at_y = get_col(at_raw, ["map_y_m", "y_m", "easting_m", "easting (local m)"])
at_z = choose_depth_column(at_raw)

fl_keep = [fl_x, fl_y, fl_z] + [c for c in ["range_m", "x_boat_m"] if c in fl_raw.columns]
at_keep = [at_x, at_y, at_z]

fl = fl_raw[fl_keep].dropna().copy()
at = at_raw[at_keep].dropna().copy()

fl = fl.rename(columns={fl_x: "x", fl_y: "y", fl_z: "z"})
at = at.rename(columns={at_x: "x", at_y: "y", at_z: "z"})

# Ensure negative altitude convention
if fl["z"].median() > 0.2:
    fl["z"] = -fl["z"]

if at["z"].median() > 0.2:
    at["z"] = -at["z"]

# -----------------------------
# IMPORTANT:
# Use reconstructed Northing as forward coordinate
# This matches your top-view mission range 40–85 m.
# -----------------------------
fl["forward_m"] = fl["x"]
at["forward_m"] = at["x"]

# Crop same section as your top-view plot
fl = fl[(fl["forward_m"] >= X_MIN) & (fl["forward_m"] <= X_MAX)].copy()
at = at[(at["forward_m"] >= X_MIN) & (at["forward_m"] <= X_MAX)].copy()

# Remove near-surface/noise
fl = fl[fl["z"] <= -0.2].copy()
at = at[at["z"] <= -0.2].copy()

print("FL points after crop:", len(fl))
print("AT points after crop:", len(at))
print("FL x range:", fl["forward_m"].min(), "to", fl["forward_m"].max())
print("AT x range:", at["forward_m"].min(), "to", at["forward_m"].max())
print("FL z range:", fl["z"].min(), "to", fl["z"].max())
print("AT z range:", at["z"].min(), "to", at["z"].max())

# -----------------------------
# DEDUPLICATE FORWARD-LOOKING
# avoids repeated observations of same forward-seen point
# -----------------------------
fl["x_bin"] = np.floor(fl["x"] / spatial_bin) * spatial_bin
fl["y_bin"] = np.floor(fl["y"] / spatial_bin) * spatial_bin

if "range_m" in fl.columns:
    sort_col = "range_m"
elif "x_boat_m" in fl.columns:
    sort_col = "x_boat_m"
else:
    sort_col = "forward_m"

fl_dedup = (
    fl.sort_values(sort_col)
      .groupby(["x_bin", "y_bin"], as_index=False)
      .first()
)

# -----------------------------
# FORWARD-LOOKING OBSTACLE PROFILE
# NO MEDIAN:
# remove only very shallow false returns, then use max(z)
# -----------------------------

fl_dedup = fl_dedup[fl_dedup["z"] <= -0.8].copy()

fl_dedup["fwd_bin"] = np.floor(fl_dedup["forward_m"] / profile_bin) * profile_bin

fl_profile = (
    fl_dedup.groupby("fwd_bin")
    .agg(
        fl_obstacle_depth=("z", lambda x: np.percentile(x,80)),
        count=("z","count")
    )
    .reset_index()
)

fl_profile = fl_profile[fl_profile["count"] >= 1].copy()

# -----------------------------
# ACROSS-TRACK MATCHED TO FL-SEEN DEPTH
# NO MEDIAN:
# For each FL bin, choose AT point closest to FL obstacle depth
# -----------------------------
at["fwd_bin"] = np.floor(at["forward_m"] / profile_bin) * profile_bin

rows = []

for _, r in fl_profile.iterrows():
    b = r["fwd_bin"]
    target_z = r["fl_obstacle_depth"]

    g = at[at["fwd_bin"] == b].copy()
    if len(g) == 0:
        continue

    g["depth_error"] = np.abs(g["z"] - target_z)
    g = g[g["depth_error"] <= depth_match_tol]

    if len(g) == 0:
        continue

    best = g.sort_values("depth_error").iloc[0]

    rows.append({
        "fwd_bin": b,
        "fl_obstacle_depth": target_z,
        "at_matched_depth": best["z"],
        "depth_error": best["depth_error"]
    })

matched = pd.DataFrame(rows)

print("FL dedup points:", len(fl_dedup))
print("FL profile points:", len(fl_profile))
print("Matched AT points:", len(matched))

if len(matched) > 0:
    diff = matched["fl_obstacle_depth"] - matched["at_matched_depth"]
    print("MAE:", np.mean(np.abs(diff)))
    print("RMSE:", np.sqrt(np.mean(diff**2)))
    print("Bias FL-AT:", np.mean(diff))

# -----------------------------
# PLOT SIDE VIEW
# -----------------------------
plt.figure(figsize=(12, 5))

plt.scatter(
    fl_dedup["forward_m"],
    fl_dedup["z"],
    s=4,
    alpha=0.20,
    color="darkorange",
    label="FL hit points"
)

plt.plot(
    fl_profile["fwd_bin"],
    fl_profile["fl_obstacle_depth"],
    lw=2.2,
    color="darkorange",
    label="Forward-looking 70° obstacle profile"
)

if len(matched) > 0:
    plt.plot(
        matched["fwd_bin"],
        matched["at_matched_depth"],
        lw=2.0,
        color="steelblue",
        label="Across-track points close to FL-seen depths"
    )

plt.xlabel("Reconstructed Northing / forward mission coordinate (m)")
plt.ylabel("Altitude relative to sonar (m)")
plt.title("Obstacle side-view profile from reconstructed point clouds")
plt.xlim(X_MIN, X_MAX)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# RESIDUAL
# -----------------------------
if len(matched) > 0:
    diff = matched["fl_obstacle_depth"] - matched["at_matched_depth"]

    plt.figure(figsize=(12, 4))
    plt.axhline(0, color="black", linestyle="--", lw=0.8)
    plt.plot(matched["fwd_bin"], diff, color="purple", lw=1.5)
    plt.fill_between(matched["fwd_bin"], diff, 0, alpha=0.25, color="purple")
    plt.xlabel("Reconstructed Northing / forward mission coordinate (m)")
    plt.ylabel("Residual FL - AT (m)")
    plt.title("Residual between FL obstacle profile and matched AT depth")
    plt.xlim(X_MIN, X_MAX)
    plt.grid(True)
    plt.tight_layout()
    plt.show()