#!/usr/bin/env python3

import math
import csv
import sys
from pathlib import Path

from brping import Surveyor240, definitions


LOG_FILE = "/home/safayat/Documents/SonarView Log Files/Surveyor 240-16/2026-03-25-18-31"
CSV_OUT = "/tmp/cerulean_offline_points.csv"

SPEED_OF_SOUND = 1500.0  # m/s, fallback if not available from packet


def atof_to_xyz(atof_points, speed_of_sound=SPEED_OF_SOUND):
    pts = []
    for p in atof_points:
        angle = float(p.angle)
        tof = float(p.tof)
        distance = 0.5 * speed_of_sound * tof

        # current local frame convention
        x = 0.0
        y = distance * math.sin(angle)
        z = -distance * math.cos(angle)

        pts.append((x, y, z))
    return pts


def main():
    log_path = Path(LOG_FILE)
    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        sys.exit(1)

    total_packets = 0
    atof_packets = 0
    yz_packets = 0
    attitude_packets = 0
    all_points = []

    with open(log_path, "rb") as f:
        while True:
            try:
                data = Surveyor240.read_packet(f)
            except Exception as e:
                print(f"Stopped reading at packet {total_packets}: {e}")
                break

            if data is None:
                break

            total_packets += 1

            try:
                mid = data.message_id
            except Exception:
                continue

            if mid == definitions.SURVEYOR240_ATOF_POINT_DATA:
                atof_packets += 1
                try:
                    atof_data = Surveyor240.create_atof_list(data)
                    pts = atof_to_xyz(atof_data)
                    all_points.extend(pts)

                    if atof_packets <= 3:
                        print(f"ATOF packet {atof_packets}: {len(pts)} points")
                        print(pts[:5])
                except Exception as e:
                    print(f"Failed to parse ATOF packet {atof_packets}: {e}")

            elif mid == definitions.SURVEYOR240_YZ_POINT_DATA:
                yz_packets += 1

            elif mid == definitions.SURVEYOR240_ATTITUDE_REPORT:
                attitude_packets += 1

    print("\nDone.")
    print(f"Total packets:   {total_packets}")
    print(f"ATOF packets:    {atof_packets}")
    print(f"YZ packets:      {yz_packets}")
    print(f"Attitude packets:{attitude_packets}")
    print(f"Total XYZ points:{len(all_points)}")

    with open(CSV_OUT, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["x", "y", "z"])
        writer.writerows(all_points)

    print(f"Saved CSV to: {CSV_OUT}")


if __name__ == "__main__":
    main()
