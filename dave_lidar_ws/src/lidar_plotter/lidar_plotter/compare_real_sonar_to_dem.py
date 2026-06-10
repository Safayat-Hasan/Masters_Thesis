import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PING_NUMBER = 39789

sonar = pd.read_csv("surveyor_atof.csv")
lidar = pd.read_csv("lidar_scan_points.csv")

ping = sonar[sonar["ping_number"] == PING_NUMBER].copy()
ping = ping.sort_values("angle_deg")
lidar = lidar.sort_values("theta_deg")

rows = []

for _, s in ping.iterrows():
    idx = (lidar["theta_deg"] - s["angle_deg"]).abs().idxmin()
    l = lidar.loc[idx]

    rows.append({
        "angle_deg": s["angle_deg"],
        "real_sonar_range_m": s["range_m"],
        "gazebo_lidar_range_m": l["range_m"],
        "angle_error_deg": l["theta_deg"] - s["angle_deg"],
        "range_error_m": s["range_m"] - l["range_m"]
    })

out = pd.DataFrame(rows)
out.to_csv("real_sonar_vs_gazebo_lidar.csv", index=False)

print(out)

print("\nMetrics:")
print("MAE:", out["range_error_m"].abs().mean())
print("RMSE:", np.sqrt((out["range_error_m"] ** 2).mean()))
print("Max error:", out["range_error_m"].abs().max())

plt.figure()
plt.plot(out["angle_deg"], out["real_sonar_range_m"], "o-", label="Real sonar")
plt.plot(out["angle_deg"], out["gazebo_lidar_range_m"], "x-", label="Gazebo LiDAR")
plt.xlabel("Beam angle (deg)")
plt.ylabel("Range (m)")
plt.title("Real Sonar vs Gazebo LiDAR Range")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()