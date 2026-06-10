import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("2026-04-22-15-40.csv")

x = df["easting (local m)"]
y = df["northing (local m)"]
z = df["altitude (m)"]

# First and last ping
first_ping = df["ping number"].min()
last_ping = df["ping number"].max()

start = df[df["ping number"] == first_ping]
end = df[df["ping number"] == last_ping]

# Use median of all detections in that ping = swath center estimate
start_x = start["easting (local m)"].median()
start_y = start["northing (local m)"].median()

end_x = end["easting (local m)"].median()
end_y = end["northing (local m)"].median()

plt.figure(figsize=(8,6))
sc = plt.scatter(x, y, c=z, s=2)

plt.scatter(start_x, start_y, color="red", s=120, marker="o", label="Start")
plt.scatter(end_x, end_y, color="black", s=120, marker="X", label="End")

plt.text(start_x, start_y, " START", color="red", fontsize=10)
plt.text(end_x, end_y, " END", color="black", fontsize=10)

plt.xlabel("Easting (local m)")
plt.ylabel("Northing (local m)")
plt.colorbar(sc, label="Depth (m)")
plt.axis("equal")
plt.legend()
plt.title("Bathymetry Map (From SonarView)")
plt.show()

# Shift origin to first point
#x0 = x[0]
#y0 = y[0]

#x_local = x - x0
#y_local = y - y0
#y_local, c=z, s=2


