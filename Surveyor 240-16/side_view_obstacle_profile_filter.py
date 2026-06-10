import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================= USER SETTINGS =================
ACROSS_ZIP = Path("2026-04-22-15-40.zip")   # across-track dataset
ALONG_ZIP  = Path("2026-04-22-16-15.zip")   # along-track / forward-looking dataset

# If you run this from another folder, put full paths, for example:
# ACROSS_ZIP = Path("/mnt/data/2026-04-22-15-40.zip")
# ALONG_ZIP  = Path("/mnt/data/2026-04-22-16-15.zip")

FORWARD_CSV_NAME = "Reconstructed_forward_pointcloud.csv"
ACROSS_CSV_NAME  = "Reconstructed_across_track_pointcloud.csv"

# repeated-return filtering for along-track data
DX_ALONG_BIN = 0.25          # metre bin along travelled direction
DEPTH_BIN = 0.10             # metre bin for depth grouping
PROFILE_METHOD = "deepest"   # "deepest", "median", or "nearest"

# across-track plot filtering
DEPTH_TOL = 0.20             # across-track points within +/- this depth of along-track profile are kept
XY_MATCH_TOL = 0.75          # across-track points must be this close in travelled-distance bin

# general cleaning
MIN_RANGE = 0.30             # remove near-field/self hits
MAX_RANGE = 8.00             # remove unrealistic far hits for this test
REMOVE_ABOVE_SURFACE = True  # keep only negative depths if True
OUTDIR = Path("profile_outputs")
# =================================================


def read_csv_from_zip(zip_path: Path, target_name: str) -> pd.DataFrame:
    """Read target CSV from a zip even if it is inside a folder."""
    with zipfile.ZipFile(zip_path) as z:
        matches = [n for n in z.namelist() if n.endswith(target_name)]
        if not matches:
            raise FileNotFoundError(f"Could not find {target_name} inside {zip_path}")
        with z.open(matches[0]) as f:
            return pd.read_csv(f)


def add_along_track_coordinate(df: pd.DataFrame, x_col="map_x_m", y_col="map_y_m") -> pd.DataFrame:
    """
    Create a 1-D travelled-distance coordinate from map_x/map_y using PCA.
    This avoids assuming the mission line is perfectly north/east aligned.
    """
    out = df.copy()
    xy = out[[x_col, y_col]].to_numpy(float)
    xy0 = xy - np.nanmean(xy, axis=0)
    _, _, vh = np.linalg.svd(xy0, full_matrices=False)
    direction = vh[0]
    s = xy0 @ direction
    out["along_track_m"] = s - np.nanmin(s)
    return out


def clean_points(df: pd.DataFrame, is_forward: bool) -> pd.DataFrame:
    out = df.copy()

    # Use global reconstructed depth. In these files this is negative downward seabed-like altitude.
    out["depth_m"] = out["map_z_m"]

    needed = ["map_x_m", "map_y_m", "depth_m", "range_m"]
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=needed)

    out = out[(out["range_m"] >= MIN_RANGE) & (out["range_m"] <= MAX_RANGE)]

    if REMOVE_ABOVE_SURFACE:
        out = out[out["depth_m"] < 0.0]

    # For forward-looking data, remove side leakage if local y exists.
    # Your forward cloud has y_boat_m very small, but this keeps the profile narrow and clean.
    if is_forward and "y_boat_m" in out.columns:
        out = out[out["y_boat_m"].abs() < 0.75]

    return add_along_track_coordinate(out)


def build_along_profile(forward: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse repeated observations of the same seabed/obstacle area.
    Instead of plotting every repeated hit, bin by along-track distance and choose one representative depth.
    """
    f = forward.copy()
    f["s_bin"] = np.round(f["along_track_m"] / DX_ALONG_BIN) * DX_ALONG_BIN
    f["z_bin"] = np.round(f["depth_m"] / DEPTH_BIN) * DEPTH_BIN

    if PROFILE_METHOD == "deepest":
        # For obstacle/terrain following, the lowest return in each distance bin gives the conservative profile.
        idx = f.groupby("s_bin")["depth_m"].idxmin()
        prof = f.loc[idx].copy()
    elif PROFILE_METHOD == "median":
        prof = (
            f.groupby("s_bin", as_index=False)
             .agg(depth_m=("depth_m", "median"), map_x_m=("map_x_m", "median"), map_y_m=("map_y_m", "median"), range_m=("range_m", "median"))
        )
    elif PROFILE_METHOD == "nearest":
        # Pick the closest/strongest-looking representative in each along bin.
        idx = f.groupby("s_bin")["range_m"].idxmin()
        prof = f.loc[idx].copy()
    else:
        raise ValueError("PROFILE_METHOD must be 'deepest', 'median', or 'nearest'")

    prof = prof.sort_values("s_bin")
    prof["profile_depth_m"] = prof["depth_m"]
    return prof


def filter_across_near_along_depth(across: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    """
    Keep across-track points only where their travelled-distance bin and depth are close
    to what the along-track sonar saw.
    """
    a = across.copy()
    a["s_bin"] = np.round(a["along_track_m"] / DX_ALONG_BIN) * DX_ALONG_BIN

    p = profile[["s_bin", "profile_depth_m"]].drop_duplicates("s_bin").sort_values("s_bin")
    a = a.sort_values("s_bin")

    matched = pd.merge_asof(
        a,
        p,
        on="s_bin",
        direction="nearest",
        tolerance=XY_MATCH_TOL,
    )
    matched = matched.dropna(subset=["profile_depth_m"])
    matched["depth_error_to_along_m"] = matched["depth_m"] - matched["profile_depth_m"]

    return matched[matched["depth_error_to_along_m"].abs() <= DEPTH_TOL].copy()


def main():
    OUTDIR.mkdir(exist_ok=True)

    across_raw = read_csv_from_zip(ACROSS_ZIP, ACROSS_CSV_NAME)
    forward_raw = read_csv_from_zip(ALONG_ZIP, FORWARD_CSV_NAME)

    across = clean_points(across_raw, is_forward=False)
    forward = clean_points(forward_raw, is_forward=True)

    profile = build_along_profile(forward)
    across_near = filter_across_near_along_depth(across, profile)

    # Save cleaned outputs
    forward.to_csv(OUTDIR / "along_track_clean_all_points.csv", index=False)
    profile.to_csv(OUTDIR / "along_track_filtered_side_profile.csv", index=False)
    across_near.to_csv(OUTDIR / "across_track_points_near_along_depth.csv", index=False)

    print(f"Forward raw points: {len(forward_raw)}")
    print(f"Forward clean points: {len(forward)}")
    print(f"Forward filtered profile points after repeated-value removal: {len(profile)}")
    print(f"Across raw points: {len(across_raw)}")
    print(f"Across points close to along-track depth profile: {len(across_near)}")

    # Plot 1: along-track raw repeated hits vs filtered profile
    plt.figure(figsize=(11, 5))
    plt.scatter(forward["along_track_m"], forward["depth_m"], s=8, alpha=0.25, label="Along-track raw repeated hits")
    plt.plot(profile["s_bin"], profile["profile_depth_m"], linewidth=2.0, label=f"Filtered side profile ({PROFILE_METHOD})")
    plt.gca().invert_yaxis()  # seabed/downward perception: more negative plotted lower visually? remove if undesired
    plt.xlabel("Along-track travelled distance (m)")
    plt.ylabel("Depth / altitude-like z (m, negative downward in this file)")
    plt.title("Along-track side-view obstacle/terrain profile after removing repeated observations")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "01_along_track_filtered_profile.png", dpi=200)

    # Plot 2: across-track points that match the depth seen by along-track sonar
    plt.figure(figsize=(11, 5))
    plt.scatter(across["along_track_m"], across["depth_m"], s=5, alpha=0.10, label="Across-track all cleaned points")
    plt.scatter(across_near["along_track_m"], across_near["depth_m"], s=12, alpha=0.75, label=f"Across-track near along depth ±{DEPTH_TOL} m")
    plt.plot(profile["s_bin"], profile["profile_depth_m"], linewidth=2.0, label="Along-track filtered profile")
    plt.gca().invert_yaxis()
    plt.xlabel("Along-track travelled distance (m)")
    plt.ylabel("Depth / altitude-like z (m, negative downward in this file)")
    plt.title("Across-track dataset filtered to depths observed by along-track run")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "02_across_track_near_along_depth.png", dpi=200)

    # Plot 3: top view of matched across-track points, useful sanity check
    plt.figure(figsize=(8, 7))
    plt.scatter(across["map_y_m"], across["map_x_m"], s=4, alpha=0.08, label="Across-track all")
    sc = plt.scatter(across_near["map_y_m"], across_near["map_x_m"], c=across_near["depth_m"], s=12, alpha=0.8, label="Across-track depth-matched")
    plt.colorbar(sc, label="Depth / altitude-like z (m)")
    plt.xlabel("Map Y / east-like (m)")
    plt.ylabel("Map X / north-like (m)")
    plt.title("Top-view sanity check: across-track points matching along-track depth profile")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "03_top_view_depth_matched_across.png", dpi=200)

    print(f"Saved outputs in: {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()
