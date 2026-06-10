import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("2026-04-22-17-06.csv")

x = df["easting (UTM m)"].to_numpy()
y = df["northing (UTM m)"].to_numpy()
z = df["altitude (m)"].to_numpy()

# Shift origin to first point
x0 = x[0]
y0 = y[0]

x_local = x - x0
y_local = y - y0

plt.scatter(x_local, y_local, c=z, s=2)
plt.colorbar(label="Depth (m)")
plt.xlabel("Easting (m from start)")
plt.ylabel("Northing (m from start)")
plt.axis('equal')
plt.title("Bathymetry Map (Local Coordinates)")
plt.show()
