import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# ── LOAD ──────────────────────────────────────────────────────────────────
at = pd.read_csv("2026-04-22-15-40.csv")
at = at[["easting (local m)", "northing (local m)", "altitude (m)"]].dropna().copy()
at.columns = ["x", "y", "z"]

fl = pd.read_csv("Reconstructed_forward_pointcloud.csv")
fl = fl[["map_y_m", "map_x_m", "altitude_like_sonarview_m"]].dropna().copy()
fl.columns = ["x", "y", "z"]

fl = fl[(fl["z"] < 0) & (fl["z"] > -8.0)]

# Keep only the straight mission segment (northing >= 48m)
# at = at[at["y"] >= 55.0]
# fl = fl[fl["y"] >= 55.0]

# ── START/END MARKERS ─────────────────────────────────────────────────────
# Across-track: use first/last ping number
at_raw = pd.read_csv("2026-04-22-15-40.csv")
at_start = at_raw[at_raw["ping number"] == at_raw["ping number"].min()]
at_end   = at_raw[at_raw["ping number"] == at_raw["ping number"].max()]
at_start_x, at_start_y = at_start["easting (local m)"].median(),  at_start["northing (local m)"].median()
at_end_x,   at_end_y   = at_end["easting (local m)"].median(),    at_end["northing (local m)"].median()

# Forward-looking: use first/last ping_number
fl_raw = pd.read_csv("Reconstructed_forward_pointcloud.csv")
fl_raw = fl_raw[(fl_raw["altitude_like_sonarview_m"] < 0) & (fl_raw["altitude_like_sonarview_m"] > -8.0)]
fl_start = fl_raw[fl_raw["ping_number"] == fl_raw["ping_number"].min()]
fl_end   = fl_raw[fl_raw["ping_number"] == fl_raw["ping_number"].max()]
fl_start_x, fl_start_y = fl_start["map_y_m"].median(), fl_start["map_x_m"].median()
fl_end_x,   fl_end_y   = fl_end["map_y_m"].median(),   fl_end["map_x_m"].median()

# ── GRID-BIN ──────────────────────────────────────────────────────────────
grid_res = 0.5

def grid_median(df, res):
    df = df.copy()
    df["xb"] = np.floor(df["x"] / res) * res
    df["yb"] = np.floor(df["y"] / res) * res
    return df.groupby(["xb", "yb"])["z"].median().reset_index()

at_grid = grid_median(at, grid_res)
fl_grid = grid_median(fl, grid_res)

# ── NEAREST-NEIGHBOUR MATCHING ─────────────────────────────────────────────
max_dist = grid_res * 1.5
at_xy = at_grid[["xb", "yb"]].to_numpy()
fl_xy = fl_grid[["xb", "yb"]].to_numpy()
tree  = cKDTree(at_xy)
dists, idxs = tree.query(fl_xy, k=1)
valid = dists <= max_dist

fl_matched = fl_grid[valid].copy().reset_index(drop=True)
at_matched = at_grid.iloc[idxs[valid]].reset_index(drop=True)
z_at = at_matched["z"].to_numpy()
z_fl = fl_matched["z"].to_numpy()
diff = z_fl - z_at

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

# ── HELPER: add start/end markers ─────────────────────────────────────────
def add_start_end(ax, sx, sy, ex, ey):
    ax.scatter(sx, sy, color="red",   s=120, marker="o", zorder=5, label="Start")
    ax.scatter(ex, ey, color="black", s=120, marker="X", zorder=5, label="End")
    ax.text(sx, sy, " START", color="red",   fontsize=9)
    ax.text(ex, ey, " END",   color="black", fontsize=9)

# ── PLOT 1: side-by-side top-view ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
vmin = min(at["z"].quantile(0.02), fl["z"].quantile(0.02))
vmax = max(at["z"].quantile(0.98), fl["z"].quantile(0.98))

sc0 = axes[0].scatter(at["x"], at["y"], c=at["z"], s=2, vmin=vmin, vmax=vmax, cmap="viridis")
add_start_end(axes[0], at_start_x, at_start_y, at_end_x, at_end_y)
axes[0].set_title("Across-track (ground truth)")
axes[0].set_xlabel("Easting (m)"); axes[0].set_ylabel("Northing (m)")
axes[0].axis("equal"); axes[0].legend(fontsize=8)
plt.colorbar(sc0, ax=axes[0], label="Altitude (m)")

sc1 = axes[1].scatter(fl["x"], fl["y"], c=fl["z"], s=2, vmin=vmin, vmax=vmax, cmap="viridis")
add_start_end(axes[1], fl_start_x, fl_start_y, fl_end_x, fl_end_y)
axes[1].set_title("Forward-looking reconstructed")
axes[1].set_xlabel("Easting (m)"); axes[1].set_ylabel("Northing (m)")
axes[1].axis("equal"); axes[1].legend(fontsize=8)
plt.colorbar(sc1, ax=axes[1], label="Altitude (m)")

plt.suptitle("Top-view point cloud comparison (matched colour scale)", fontsize=13)
plt.tight_layout()
plt.show()

# ── PLOT 2: depth error map ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
sc = ax.scatter(fl_matched["xb"], fl_matched["yb"],
                c=diff, cmap="RdBu", vmin=-0.5, vmax=0.5, s=12)
add_start_end(ax, fl_start_x, fl_start_y, fl_end_x, fl_end_y)
plt.colorbar(sc, ax=ax, label="Depth error: FL − AT (m)")
ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
ax.set_title(f"Depth error map   RMSE={rmse:.3f}m   Bias={bias:+.3f}m")
ax.axis("equal"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# ── PLOT 3: scatter + histogram (no spatial markers needed) ───────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(z_at, z_fl, s=4, alpha=0.3, color="steelblue")
lims = [min(z_at.min(), z_fl.min()) - 0.1, max(z_at.max(), z_fl.max()) + 0.1]
axes[0].plot(lims, lims, "k--", lw=1, label="1:1 line")
axes[0].set_xlabel("Across-track depth (m)"); axes[0].set_ylabel("Forward-looking depth (m)")
axes[0].set_title(f"Depth scatter   r={r:.3f}")
axes[0].legend(); axes[0].grid(True)
axes[0].set_xlim(lims); axes[0].set_ylim(lims); axes[0].set_aspect("equal")

axes[1].hist(diff, bins=40, color="steelblue", edgecolor="white", linewidth=0.5)
axes[1].axvline(0,    color="black",      lw=1,   linestyle="--", label="zero")
axes[1].axvline(bias, color="darkorange", lw=1.5, linestyle=":",  label=f"bias={bias:+.3f}m")
axes[1].set_xlabel("Depth error: FL − AT (m)"); axes[1].set_ylabel("Count")
axes[1].set_title(f"Error distribution   MAE={mae:.3f}m   RMSE={rmse:.3f}m")
axes[1].legend(); axes[1].grid(True)

plt.tight_layout()
plt.show()

