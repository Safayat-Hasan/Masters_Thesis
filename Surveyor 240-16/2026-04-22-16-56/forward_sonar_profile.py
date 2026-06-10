import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── FILES USED ─────────────────────────────────────────────────────────────
# 1. 2026-04-22-16-15.csv        — SonarView exported simplified CSV
#                                  (forward-looking sonar detections, geo-located)
#                                  Columns used: northing (local m), easting (local m),
#                                                altitude (m), elapsed (s), ping number
#
# 2. blueboat_local_position.csv — raw mission data, boat position over time
#                                  Columns used: time_boot_ms, x_m (northing), y_m (easting)
# ───────────────────────────────────────────────────────────────────────────

sonar = pd.read_csv("2026-04-22-16-56.csv")
pos   = pd.read_csv("blueboat_local_position.csv")

# Align position timestamps to elapsed seconds (same reference as sonar)
pos['elapsed_s'] = (pos['time_boot_ms'] - pos['time_boot_ms'].min()) / 1000.0

# Interpolate boat position at each sonar ping time
sonar = sonar.sort_values('elapsed (s)').copy()
sonar['boat_north'] = np.interp(sonar['elapsed (s)'], pos['elapsed_s'], pos['x_m'])
sonar['boat_east']  = np.interp(sonar['elapsed (s)'], pos['elapsed_s'], pos['y_m'])

# PCA on boat track to find the main forward motion axis
boat_xy = sonar[['boat_north', 'boat_east']].drop_duplicates().to_numpy()
origin  = boat_xy.mean(axis=0)
_, _, vh = np.linalg.svd(boat_xy - origin, full_matrices=False)
forward = vh[0]

# Ensure forward axis follows time direction
first = sonar.nsmallest(200, 'elapsed (s)')[['boat_north', 'boat_east']].mean().to_numpy()
last  = sonar.nlargest(200,  'elapsed (s)')[['boat_north', 'boat_east']].mean().to_numpy()
if np.dot(forward, last - first) < 0:
    forward = -forward

# Project DETECTION world position onto forward axis (X axis of the side-view plot)
det_xy = sonar[['northing (local m)', 'easting (local m)']].to_numpy()
sonar['det_forward_m'] = det_xy @ forward
sonar['z_m'] = sonar['altitude (m)']

# Range from boat to each detection (used for deduplication and filtering)
sonar['range_m'] = np.sqrt(
    (sonar['northing (local m)'] - sonar['boat_north'])**2 +
    (sonar['easting (local m)']  - sonar['boat_east'])**2
)

# ── FILTER 1: spatial crop — plot from 60m forward position onward
sonar = sonar[(sonar['det_forward_m'] >= -22.0) & (sonar['det_forward_m'] <= -5.0)]

# ── FILTER 2: drop near-zero altitude returns (surface / noise reflections)
# Real seabed is at least 0.5m below sonar; anything shallower is a false return.
# Adjust -0.5 if your sonar is mounted higher or lower.
sonar = sonar[sonar['z_m'] <= -0.5]

# ── FILTER 3: drop very short range detections (directly-under-hull noise)
sonar = sonar[sonar['range_m'] >= 0.5]

# ── FILTER 4: IQR outlier removal per 1m forward bin (multiplier = 1.0, strict)
# Removes remaining outliers that survive the hard cuts above.
# Tune bin_size_iqr (larger = more global) and multiplier (smaller = stricter).
bin_size_iqr = 1.0
sonar['iqr_bin'] = np.floor(sonar['det_forward_m'] / bin_size_iqr) * bin_size_iqr

def iqr_filter(g):
    q1, q3 = g['z_m'].quantile(0.25), g['z_m'].quantile(0.75)
    iqr = q3 - q1
    return g[(g['z_m'] >= q1 - 1.0 * iqr) & (g['z_m'] <= q3 + 1.0 * iqr)]

sonar = sonar.groupby('iqr_bin', group_keys=False).apply(iqr_filter).reset_index(drop=True)

# ── DEDUPLICATE ─────────────────────────────────────────────────────────────
# A forward-looking sonar sees the same world point repeatedly as the boat
# approaches. Bin by world-space position, keep closest-range observation
# (least angular distortion, most accurate altitude).
bin_size = 0.25  # metres — tune to your sonar resolution

sonar['n_bin'] = np.floor(sonar['northing (local m)'] / bin_size) * bin_size
sonar['e_bin'] = np.floor(sonar['easting (local m)']  / bin_size) * bin_size

dedup = (
    sonar.sort_values('range_m')
         .groupby(['n_bin', 'e_bin'], as_index=False)
         .first()   # closest-range observation per world-space cell
)

# ── PROFILE ─────────────────────────────────────────────────────────────────
profile_bin = 0.25
dedup['fwd_bin'] = np.floor(dedup['det_forward_m'] / profile_bin) * profile_bin
profile = dedup.groupby('fwd_bin')['z_m'].agg(
    median_depth='median',
    deepest_point='min',
    count='count'
).reset_index()
profile = profile[profile['count'] >= 2]

# ── PLOT ─────────────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 5))
plt.scatter(dedup['det_forward_m'], dedup['z_m'], s=3, alpha=0.25, color='steelblue')
plt.plot(profile['fwd_bin'], profile['median_depth'],  lw=2,   color='steelblue',  label='median profile')
plt.plot(profile['fwd_bin'], profile['deepest_point'], lw=1.5, color='darkorange', linestyle='--', label='deepest envelope')
plt.xlabel('Projected detection position along forward motion (m)')
plt.ylabel('Altitude relative to sonar (m)')
plt.title('Along-track 50° Forward-looking sonar — side-view profile towards an obstacle(Rock)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
