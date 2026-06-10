import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# ── LOAD ──────────────────────────────────────────────────────────────────
# Across-track: SonarView CSV (ground truth)
at = pd.read_csv("2026-04-22-15-40.csv")
at = at[["easting (local m)", "northing (local m)", "altitude (m)"]].dropna().copy()
at.columns = ["x", "y", "z"]

# Forward-looking: reconstructed point cloud
fl = pd.read_csv("Reconstructed_across_track_pointcloud.csv")
fl = fl[["map_y_m", "map_x_m", "altitude_like_sonarview_m"]].dropna().copy()
fl.columns = ["x", "y", "z"]

# Remove physically impossible altitudes and spin-up artefacts
fl = fl[fl["z"] < 0]        # must be below sonar
fl = fl[fl["z"] > -8.0]     # must be within sonar range

# Keep only the straight mission segment (northing >= 48m)
# at = at[at["y"] >= 55.0]
# fl = fl[fl["y"] >= 55.0]

# ── GRID-BIN both clouds to a common resolution ───────────────────────────
# Avoids density bias: one depth value per cell for each cloud
# then compare cells that both cover
grid_res = 0.5  # metres — tune to your sonar resolution

def grid_median(df, res):
    df = df.copy()
    df["xb"] = np.floor(df["x"] / res) * res
    df["yb"] = np.floor(df["y"] / res) * res
    return df.groupby(["xb", "yb"])["z"].median().reset_index()

at_grid = grid_median(at, grid_res)
fl_grid = grid_median(fl, grid_res)

# ── NEAREST-NEIGHBOUR MATCHING ─────────────────────────────────────────────
# For each forward-looking grid cell, find closest across-track cell
# Only accept matches within max_dist (avoids matching across gaps)
max_dist = grid_res * 1.5  # metres

at_xy  = at_grid[["xb", "yb"]].to_numpy()
fl_xy  = fl_grid[["xb", "yb"]].to_numpy()

tree   = cKDTree(at_xy)
dists, idxs = tree.query(fl_xy, k=1)

valid  = dists <= max_dist
fl_matched = fl_grid[valid].copy().reset_index(drop=True)
at_matched = at_grid.iloc[idxs[valid]].reset_index(drop=True)

z_at   = at_matched["z"].to_numpy()
z_fl   = fl_matched["z"].to_numpy()
diff   = z_fl - z_at

# ── METRICS ───────────────────────────────────────────────────────────────
rmse     = np.sqrt(np.mean(diff**2))
mae      = np.mean(np.abs(diff))
bias     = np.mean(diff)
r        = np.corrcoef(z_at, z_fl)[0, 1]
rmse_deb = np.sqrt(np.mean((diff - bias)**2))
coverage = valid.sum() / len(at_grid) * 100

print("─── Point cloud comparison: forward-looking vs across-track ───")
print(f"  Grid resolution   : {grid_res} m")
print(f"  Across-track cells: {len(at_grid)}")
print(f"  Forward-look cells: {len(fl_grid)}")
print(f"  Matched pairs     : {valid.sum()}  ({coverage:.1f}% of across-track covered)")
print(f"  RMSE              : {rmse:.4f} m")
print(f"  MAE               : {mae:.4f} m")
print(f"  Bias              : {bias:+.4f} m  ({'FL shallower' if bias > 0 else 'FL deeper'})")
print(f"  Pearson r         : {r:.4f}")
print(f"  RMSE (bias-removed): {rmse_deb:.4f} m")
print(f"  z_at std          : {z_at.std():.4f} m")
print(f"  z_fl std          : {z_fl.std():.4f} m")
print(f"  depth range (AT)  : {z_at.min():.3f} to {z_at.max():.3f} m")

# ── PLOT 1: side-by-side top-view coloured by depth ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
vmin = min(at["z"].quantile(0.02), fl["z"].quantile(0.02))
vmax = max(at["z"].quantile(0.98), fl["z"].quantile(0.98))

sc0 = axes[0].scatter(at["x"], at["y"], c=at["z"], s=2, vmin=vmin, vmax=vmax, cmap="viridis")
axes[0].set_title("Across-track (ground truth)")
axes[0].set_xlabel("Easting (m)")
axes[0].set_ylabel("Northing (m)")
axes[0].axis("equal")
plt.colorbar(sc0, ax=axes[0], label="Altitude (m)")

sc1 = axes[1].scatter(fl["x"], fl["y"], c=fl["z"], s=2, vmin=vmin, vmax=vmax, cmap="viridis")
axes[1].set_title("Forward-looking reconstructed")
axes[1].set_xlabel("Easting (m)")
axes[1].set_ylabel("Northing (m)")
axes[1].axis("equal")
plt.colorbar(sc1, ax=axes[1], label="Altitude (m)")

plt.suptitle("Top-view point cloud comparison (matched colour scale)", fontsize=13)
plt.tight_layout()
plt.show()

# ── PLOT 2: depth error map ────────────────────────────────────────────────
# Spatial map of where the forward-looker agrees / disagrees with across-track
fig, ax = plt.subplots(figsize=(8, 7))
sc = ax.scatter(
    fl_matched["xb"], fl_matched["yb"],
    c=diff, cmap="RdBu", vmin=-0.5, vmax=0.5, s=12
)
plt.colorbar(sc, ax=ax, label="Depth error: FL − AT (m)")
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")
ax.set_title(f"Depth error map   RMSE={rmse:.3f}m   Bias={bias:+.3f}m")
ax.axis("equal")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ── PLOT 3: scatter plot of matched depths ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(z_at, z_fl, s=4, alpha=0.3, color="steelblue")
lims = [min(z_at.min(), z_fl.min()) - 0.1, max(z_at.max(), z_fl.max()) + 0.1]
axes[0].plot(lims, lims, "k--", lw=1, label="1:1 line")
axes[0].set_xlabel("Across-track depth (m)")
axes[0].set_ylabel("Forward-looking depth (m)")
axes[0].set_title(f"Depth scatter   r={r:.3f}")
axes[0].legend()
axes[0].grid(True)
axes[0].set_xlim(lims); axes[0].set_ylim(lims)
axes[0].set_aspect("equal")

axes[1].hist(diff, bins=40, color="steelblue", edgecolor="white", linewidth=0.5)
axes[1].axvline(0,    color="black",      lw=1,   linestyle="--", label="zero")
axes[1].axvline(bias, color="darkorange", lw=1.5, linestyle=":",  label=f"bias={bias:+.3f}m")
axes[1].set_xlabel("Depth error: FL − AT (m)")
axes[1].set_ylabel("Count")
axes[1].set_title(f"Error distribution   MAE={mae:.3f}m   RMSE={rmse:.3f}m")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()