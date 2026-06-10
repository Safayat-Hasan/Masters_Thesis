import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# ── PIPELINE 1: SonarView CSV ─────────────────────────────────────────────
df_sv = pd.read_csv("2026-04-22-15-40.csv")
d = df_sv[["northing (local m)", "easting (local m)", "altitude (m)", "elapsed (s)"]].dropna().copy()

XY = d[["northing (local m)", "easting (local m)"]].to_numpy()
origin = XY.mean(axis=0)
_, _, vh = np.linalg.svd(XY - origin, full_matrices=False)
forward = vh[0]
first = d.nsmallest(200, "elapsed (s)")[["northing (local m)", "easting (local m)"]].mean().to_numpy()
last  = d.nlargest(200,  "elapsed (s)")[["northing (local m)", "easting (local m)"]].mean().to_numpy()
if np.dot(forward, last - first) < 0:
    forward = -forward

d["forward_m"] = XY @ forward
d["z_m"] = d["altitude (m)"]
d = d[d["forward_m"] >= 60.0]

bin_size = 0.25
d["bin"] = np.floor(d["forward_m"] / bin_size) * bin_size
profile_sv = d.groupby("bin")["z_m"].agg(median_depth="median", count="count").reset_index()
profile_sv = profile_sv[profile_sv["count"] >= 3]

# ── PIPELINE 2: Reconstructed CSV ────────────────────────────────────────
sonar    = pd.read_csv("surveyor_atof.csv")
boat_pos = pd.read_csv("blueboat_local_position.csv")
boat_att = pd.read_csv("blueboat_attitude.csv")

boat_pos = boat_pos.rename(columns={"x_m": "boat_x_m", "y_m": "boat_y_m", "z_m": "boat_z_m"})
time_offset_ms = sonar["pwr_up_msec"].min() - boat_pos["time_boot_ms"].min()
sonar["t_sync"]    = sonar["pwr_up_msec"]
boat_pos["t_sync"] = boat_pos["time_boot_ms"] + time_offset_ms
boat_att["t_sync"] = boat_att["time_boot_ms"] + time_offset_ms

sonar    = sonar.sort_values("t_sync")
boat_pos = boat_pos.sort_values("t_sync")
boat_att = boat_att.sort_values("t_sync")

merged = pd.merge_asof(sonar,    boat_pos[["t_sync", "boat_x_m", "boat_y_m", "boat_z_m"]], on="t_sync", direction="nearest", tolerance=100)
merged = pd.merge_asof(merged,   boat_att[["t_sync", "roll_rad", "pitch_rad", "yaw_rad"]],  on="t_sync", direction="nearest", tolerance=100)
merged = merged.dropna().copy()

p_sonar = np.vstack([
    np.zeros(len(merged)),
    -merged["y_m"].to_numpy(),
     merged["z_m"].to_numpy()
]).T

mount_rot = R.from_euler("xyz", [-2.72, 2.78, 0.0], degrees=True)
sonar_offset_boat = np.array([0.0, 0.16, 0.14])
p_world = []
for k, row in enumerate(merged.itertuples(index=False)):
    boat_rot = R.from_euler("xyz", [row.roll_rad, row.pitch_rad, row.yaw_rad], degrees=False)
    p_boat = mount_rot.apply(p_sonar[k]) + sonar_offset_boat
    p = boat_rot.apply(p_boat)
    p[0] += row.boat_x_m
    p[1] += row.boat_y_m
    p_world.append(p)

p_world = np.array(p_world)
merged["map_x_m"] = p_world[:, 0]
merged["map_y_m"] = p_world[:, 1]
merged["map_z_m"] = p_world[:, 2]
merged["altitude_like_sonarview_m"] = merged["map_z_m"]

df_rc = merged.copy()
XY2 = df_rc[["map_x_m", "map_y_m"]].to_numpy()
origin2 = XY2.mean(axis=0)
_, _, vh2 = np.linalg.svd(XY2 - origin2, full_matrices=False)
forward2 = vh2[0]
first2 = df_rc.nsmallest(200, "t_sync")[["map_x_m", "map_y_m"]].mean().to_numpy()
last2  = df_rc.nlargest(200,  "t_sync")[["map_x_m", "map_y_m"]].mean().to_numpy()
if np.dot(forward2, last2 - first2) < 0:
    forward2 = -forward2

df_rc["forward_m"] = XY2 @ forward2
df_rc["z_m"] = df_rc["altitude_like_sonarview_m"]
df_rc = df_rc[df_rc["forward_m"] >= 60.0]

df_rc["bin"] = np.floor(df_rc["forward_m"] / bin_size) * bin_size
profile_rc = df_rc.groupby("bin")["z_m"].agg(median_depth="median", count="count").reset_index()
profile_rc = profile_rc[profile_rc["count"] >= 3]

# ── COMBINED PLOT ─────────────────────────────────────────────────────────
plt.figure(figsize=(12, 5))
plt.plot(profile_sv["bin"], profile_sv["median_depth"], lw=2, color="steelblue",  label="SonarView CSV (median)")
plt.plot(profile_rc["bin"], profile_rc["median_depth"], lw=2, color="darkorange", label="Reconstructed (median)")
plt.xlabel("Projected sonar detection position along forward motion (m)")
plt.ylabel("Altitude relative to sonar (m)")
plt.title("Side-view median profile comparison: SonarView vs Reconstructed")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ── Assume profile_at and profile_fl are already computed from your two pipelines
# profile_at has columns: bin, median_depth
# profile_fl has columns: fwd_bin, median_depth
# Rename for clarity
at = profile_sv.rename(columns={"bin": "x", "median_depth": "z_at"})
fl = profile_rc.rename(columns={"bin": "x", "median_depth": "z_fl"})

# ── Merge on shared X bins (inner join = only where both have data)
merged = pd.merge(at, fl, on="x", how="inner")

z_at = merged["z_at"].to_numpy()
z_fl = merged["z_fl"].to_numpy()
diff = z_fl - z_at

# ── Metrics
rmse     = np.sqrt(np.mean(diff**2))
mae      = np.mean(np.abs(diff))
bias     = np.mean(diff)
r        = np.corrcoef(z_at, z_fl)[0, 1]
rmse_deb = np.sqrt(np.mean((diff - bias)**2))  # bias-corrected RMSE
coverage = len(merged) / len(at) * 100

print("─── Quantitative comparison: forward-looking vs across-track ───")
print(f"  Matched bins      : {len(merged)}  /  {len(at)}  ({coverage:.1f}% coverage)")
print(f"  RMSE              : {rmse:.4f} m")
print(f"  MAE               : {mae:.4f} m")
print(f"  Bias              : {bias:+.4f} m  ({'forward-looker shallower' if bias > 0 else 'forward-looker deeper'})")
print(f"  Pearson r         : {r:.4f}")
print(f"  RMSE (bias-removed): {rmse_deb:.4f} m")

# Add this to your comparison script to diagnose the low r
print("z_at std:", z_at.std().round(4), "m")
print("z_fl std:", z_fl.std().round(4), "m")
print("depth range:", z_at.min().round(3), "to", z_at.max().round(3), "m")

# ── Residual plot
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

axes[0].plot(merged["x"], z_at, color="steelblue",  lw=2, label="Across-track (ground truth)")
axes[0].plot(merged["x"], z_fl, color="darkorange", lw=2, label="Forward-looking")
axes[0].set_ylabel("Altitude (m)")
axes[0].set_title("Profile comparison (matched bins only)")
axes[0].legend()
axes[0].grid(True)

axes[1].axhline(0,     color="black",     lw=0.8, linestyle="--")
axes[1].axhline(bias,  color="darkorange", lw=1.2, linestyle=":", label=f"Bias = {bias:+.3f} m")
axes[1].fill_between(merged["x"], diff, 0, alpha=0.25, color="darkorange")
axes[1].plot(merged["x"], diff, color="darkorange", lw=1.5)
axes[1].set_ylabel("Residual: FL − AT (m)")
axes[1].set_xlabel("Forward position (m)")
axes[1].set_title(f"Residuals   RMSE={rmse:.3f} m   MAE={mae:.3f} m   r={r:.3f}")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()