# import pandas as pd
# import numpy as np
# from scipy.interpolate import griddata
# import matplotlib.pyplot as plt
# import imageio.v2 as imageio
# from scipy.spatial import Delaunay


# # =========================
# # LOAD CSV
# # =========================

# df = pd.read_csv("Reconstructed_across_track_pointcloud.csv")

# # World coordinates
# x = df["map_x_m"].to_numpy()
# y = df["map_y_m"].to_numpy()

# # Gazebo prefers z-up
# # Your seabed depths are negative
# z = df["map_z_m"].to_numpy()

# # =========================
# # CREATE REGULAR GRID
# # =========================

# N = 513   # 2^n + 1 (Gazebo friendly)

# xi = np.linspace(x.min(), x.max(), N)
# yi = np.linspace(y.min(), y.max(), N)

# X, Y = np.meshgrid(xi, yi)

# # =========================
# # INTERPOLATE TERRAIN
# # =========================

# Z = griddata(
#     (x, y),
#     z,
#     (X, Y),
#     method='nearest'
# )

# points = np.vstack((x,y)).T

# tri = Delaunay(points)

# mask = tri.find_simplex(
#     np.vstack((X.flatten(), Y.flatten())).T
# )

# mask = mask.reshape(X.shape)

# Z[mask < 0] = np.nan

# # # Fill missing holes
# # Z_nearest = griddata(
# #     (x, y),
# #     z,
# #     (X, Y),
# #     method='nearest'
# # )

# #Z[np.isnan(Z)] = Z_nearest[np.isnan(Z)]

# # =========================
# # VISUALIZE DEM
# # =========================

# plt.figure(figsize=(10,6))
# plt.imshow(
#     Z,
#     extent=[x.min(), x.max(), y.min(), y.max()],
#     origin='lower'
# )
# plt.colorbar(label='Depth (m)')
# plt.title("Generated DEM")
# plt.xlabel("X (m)")
# plt.ylabel("Y (m)")
# plt.axis('equal')
# plt.show()

# # =========================
# # NORMALIZE FOR HEIGHTMAP
# # =========================

# valid = ~np.isnan(Z)

# rows = np.any(valid, axis=1)
# cols = np.any(valid, axis=0)

# Z_crop = Z[np.ix_(rows, cols)]

# z_min = np.nanmin(Z_crop)
# z_max = np.nanmax(Z_crop)

# Z_norm = np.zeros_like(Z_crop, dtype=np.uint8)

# valid_crop = ~np.isnan(Z_crop)

# Z_norm[valid_crop] = (
#     255 * (Z_crop[valid_crop] - z_min)
#     / (z_max - z_min)
# ).astype(np.uint8)

# imageio.imwrite("seabed_heightmap.png", Z_norm)

# print("Saved seabed_heightmap.png")
# print("z_min:", z_min)
# print("z_max:", z_max)

# plt.figure(figsize=(8,8))
# plt.imshow(Z_norm, cmap="gray", origin="lower")
# plt.colorbar(label="Normalized height")
# plt.title("Saved Gazebo Heightmap")
# plt.axis("equal")
# plt.show()

# Run this script to generate a colored OBJ mesh from your DEM
# This completely bypasses the heightmap texture problem

# import pandas as pd
# import numpy as np
# from scipy.interpolate import griddata
# from PIL import Image

# # --- Load your data ---
# df = pd.read_csv('Reconstructed_across_track_pointcloud.csv')

# x = df['map_x_m'].values
# y = df['map_y_m'].values
# z = df['map_z_m'].values

# # --- Build DEM grid ---
# resolution = 0.25
# xi = np.arange(x.min(), x.max(), resolution)
# yi = np.arange(y.min(), y.max(), resolution)
# Xi, Yi = np.meshgrid(xi, yi)

# Zi = griddata((x, y), z, (Xi, Yi), method='linear')
# Zi_nn = griddata((x, y), z, (Xi, Yi), method='nearest')
# Zi[np.isnan(Zi)] = Zi_nn[np.isnan(Zi)]

# rows, cols = Zi.shape
# print(f"Grid: {rows} x {cols}, Z: {Zi.min():.3f} to {Zi.max():.3f}")

# # --- Write OBJ mesh ---
# obj_path = '/home/safayat/gazebo_worlds/seabed_dem.obj'
# mtl_path = '/home/safayat/gazebo_worlds/seabed_dem.mtl'

# with open(obj_path, 'w') as f:
#     f.write("# Seabed DEM mesh\n")
#     f.write("mtllib seabed_dem.mtl\n")

#     # Write vertices
#     for r in range(rows):
#         for c in range(cols):
#             vx = xi[c] - x.min()   # shift to origin
#             vy = yi[r] - y.min()
#             vz = Zi[r, c]
#             f.write(f"v {vx:.4f} {vy:.4f} {vz:.4f}\n")

#     # Write UVs
#     for r in range(rows):
#         for c in range(cols):
#             u = c / (cols - 1)
#             v = r / (rows - 1)
#             f.write(f"vt {u:.4f} {v:.4f}\n")

#     # Write normals (flat up for now)
#     f.write("vn 0 0 1\n")

#     f.write("usemtl seabed_mat\n")

#     # Write faces (two triangles per quad)
#     def idx(r, c): return r * cols + c + 1  # OBJ is 1-indexed

#     for r in range(rows - 1):
#         for c in range(cols - 1):
#             v0 = idx(r,   c);   v1 = idx(r,   c+1)
#             v2 = idx(r+1, c+1); v3 = idx(r+1, c)
#             f.write(f"f {v0}/{v0}/1 {v1}/{v1}/1 {v2}/{v2}/1\n")
#             f.write(f"f {v0}/{v0}/1 {v2}/{v2}/1 {v3}/{v3}/1\n")

# print(f"OBJ saved: {obj_path}")

# # --- Write MTL with sandy color ---
# with open(mtl_path, 'w') as f:
#     f.write("newmtl seabed_mat\n")
#     f.write("Ka 0.7 0.6 0.4\n")   # ambient - sandy brown
#     f.write("Kd 0.7 0.6 0.4\n")   # diffuse
#     f.write("Ks 0.1 0.1 0.1\n")   # specular
#     f.write("Ns 10\n")
#     f.write("d 1\n")

# print(f"MTL saved: {mtl_path}")

import pandas as pd
import numpy as np
from scipy.interpolate import griddata

df = pd.read_csv('/home/safayat/Downloads/OneDrive_1_4-23-2026/Surveyor 240-16/2026-04-22-15-40/Reconstructed_across_track_pointcloud.csv')

x = df['map_x_m'].values
y = df['map_y_m'].values
z = df['map_z_m'].values

resolution = 0.25
xi = np.arange(x.min(), x.max(), resolution)
yi = np.arange(y.min(), y.max(), resolution)
Xi, Yi = np.meshgrid(xi, yi)

Zi = griddata((x, y), z, (Xi, Yi), method='linear')
Zi_nn = griddata((x, y), z, (Xi, Yi), method='nearest')
Zi[np.isnan(Zi)] = Zi_nn[np.isnan(Zi)]

rows, cols = Zi.shape

obj_path = '/home/safayat/gazebo_worlds/seabed_dem.obj'
mtl_path = '/home/safayat/gazebo_worlds/seabed_dem.mtl'

with open(obj_path, 'w') as f:
    f.write("# Seabed DEM mesh\n")
    f.write("mtllib seabed_dem.mtl\n\n")

    # Vertices — shift to origin
    for r in range(rows):
        for c in range(cols):
            vx = xi[c] - xi[0]
            vy = yi[r] - yi[0]
            vz = Zi[r, c]
            f.write(f"v {vx:.4f} {vy:.4f} {vz:.4f}\n")

    # UVs
    for r in range(rows):
        for c in range(cols):
            f.write(f"vt {c/(cols-1):.4f} {r/(rows-1):.4f}\n")

    # Per-vertex normals using finite differences
    # dz/dx and dz/dy give the surface gradient → cross product = normal
    Zx = np.gradient(Zi, axis=1) / resolution  # dz/dx
    Zy = np.gradient(Zi, axis=0) / resolution  # dz/dy

    for r in range(rows):
        for c in range(cols):
            # Normal = (-dz/dx, -dz/dy, 1) normalised
            nx = -Zx[r, c]
            ny = -Zy[r, c]
            nz = 1.0
            length = np.sqrt(nx**2 + ny**2 + nz**2)
            f.write(f"vn {nx/length:.4f} {ny/length:.4f} {nz/length:.4f}\n")

    f.write("\nusemtl seabed_mat\n")

    # Faces with vertex/uv/normal indices
    def idx(r, c): return r * cols + c + 1

    for r in range(rows - 1):
        for c in range(cols - 1):
            v0=idx(r,c); v1=idx(r,c+1); v2=idx(r+1,c+1); v3=idx(r+1,c)
            f.write(f"f {v0}/{v0}/{v0} {v1}/{v1}/{v1} {v2}/{v2}/{v2}\n")
            f.write(f"f {v0}/{v0}/{v0} {v2}/{v2}/{v2} {v3}/{v3}/{v3}\n")

print(f"OBJ saved → {obj_path}  ({rows}x{cols} grid, {(rows-1)*(cols-1)*2} faces)")

with open(mtl_path, 'w') as f:
    f.write("newmtl seabed_mat\n")
    f.write("Ka 0.5 0.4 0.3\n")
    f.write("Kd 0.65 0.55  0.35\n")
    f.write("Ks 0.05 0.05 0.05\n")
    f.write("Ns 5\n")
    f.write("d 1\n")

print("MTL saved")