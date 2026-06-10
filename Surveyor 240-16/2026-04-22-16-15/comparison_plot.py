import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

bin_size = 0.25

# ── PIPELINE 1: Across-track (2026-04-22-15-40.csv) ──────────────────────
df_at = pd.read_csv("2026-04-22-15-40.csv")
d = df_at[["northing (local m)", "easting (local m)", "altitude (m)", "elapsed (s)"]].dropna().copy()

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

d["bin"] = np.floor(d["forward_m"] / bin_size) * bin_size
profile_at = d.groupby("bin")["z_m"].agg(median_depth="median", count="count").reset_index()
profile_at = profile_at[profile_at["count"] >= 3]

# ── PIPELINE 2: Forward-looking (2026-04-22-15-58.csv) ───────────────────
sonar = pd.read_csv("2026-04-22-16-15.csv")
pos   = pd.read_csv("blueboat_local_position.csv")

pos['elapsed_s'] = (pos['time_boot_ms'] - pos['time_boot_ms'].min()) / 1000.0
sonar = sonar.sort_values('elapsed (s)').copy()
sonar['boat_north'] = np.interp(sonar['elapsed (s)'], pos['elapsed_s'], pos['x_m'])
sonar['boat_east']  = np.interp(sonar['elapsed (s)'], pos['elapsed_s'], pos['y_m'])

boat_xy = sonar[['boat_north', 'boat_east']].drop_duplicates().to_numpy()
origin  = boat_xy.mean(axis=0)
_, _, vh = np.linalg.svd(boat_xy - origin, full_matrices=False)
forward = vh[0]
first = sonar.nsmallest(200, 'elapsed (s)')[['boat_north', 'boat_east']].mean().to_numpy()
last  = sonar.nlargest(200,  'elapsed (s)')[['boat_north', 'boat_east']].mean().to_numpy()
if np.dot(forward, last - first) < 0:
    forward = -forward

det_xy = sonar[['northing (local m)', 'easting (local m)']].to_numpy()
sonar['det_forward_m'] = det_xy @ forward
sonar['z_m'] = sonar['altitude (m)']
sonar['range_m'] = np.sqrt(
    (sonar['northing (local m)'] - sonar['boat_north'])**2 +
    (sonar['easting (local m)']  - sonar['boat_east'])**2
)

sonar = sonar[sonar['det_forward_m'] >= 60.0]
sonar = sonar[sonar['z_m'] <= -0.5]
sonar = sonar[sonar['range_m'] >= 0.5]

bin_size_iqr = 1.0
sonar['iqr_bin'] = np.floor(sonar['det_forward_m'] / bin_size_iqr) * bin_size_iqr
def iqr_filter(g):
    q1, q3 = g['z_m'].quantile(0.25), g['z_m'].quantile(0.75)
    iqr = q3 - q1
    return g[(g['z_m'] >= q1 - 1.0 * iqr) & (g['z_m'] <= q3 + 1.0 * iqr)]
sonar = sonar.groupby('iqr_bin', group_keys=False).apply(iqr_filter).reset_index(drop=True)

sonar['n_bin'] = np.floor(sonar['northing (local m)'] / bin_size) * bin_size
sonar['e_bin'] = np.floor(sonar['easting (local m)']  / bin_size) * bin_size
dedup = (
    sonar.sort_values('range_m')
         .groupby(['n_bin', 'e_bin'], as_index=False)
         .first()
)

dedup['fwd_bin'] = np.floor(dedup['det_forward_m'] / bin_size) * bin_size
profile_fl = dedup.groupby('fwd_bin')['z_m'].agg(median_depth='median', count='count').reset_index()
profile_fl = profile_fl[profile_fl['count'] >= 2]

# ── COMBINED PLOT ─────────────────────────────────────────────────────────
plt.figure(figsize=(12, 5))
plt.plot(profile_at["bin"],     profile_at["median_depth"], lw=2, color="steelblue",  label="Across-track")
plt.plot(profile_fl["fwd_bin"], profile_fl["median_depth"], lw=2, color="darkorange", label="Forward-looking-70")
plt.xlabel("Projected sonar detection position along forward motion (m)")
plt.ylabel("Altitude relative to sonar (m)")
plt.title("Side-view median profile comparison: Across-track vs Forward-looking-70")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ── Assume profile_at and profile_fl are already computed from your two pipelines
# profile_at has columns: bin, median_depth
# profile_fl has columns: fwd_bin, median_depth
# Rename for clarity
at = profile_at.rename(columns={"bin": "x", "median_depth": "z_at"})
fl = profile_fl.rename(columns={"fwd_bin": "x", "median_depth": "z_fl"})

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