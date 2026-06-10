#!/usr/bin/env python3

import argparse
import csv
import gzip
import json
import math
import struct
from datetime import datetime, timezone
from pathlib import Path

from brping import Surveyor240, definitions


def open_log(path):
    with open(path, "rb") as f:
        magic = f.read(2)
    return gzip.open(path, "rb") if magic == b"\x1f\x8b" else open(path, "rb")


def payload_bytes(msg):
    raw = bytes(msg.msg_data)
    payload_len = raw[2] | (raw[3] << 8)
    return raw[8:8 + payload_len]


def parse_atof_payload(payload):
    header_fmt = "<IQffIIfIHH"
    header_len = struct.calcsize(header_fmt)

    header_values = struct.unpack_from(header_fmt, payload, 0)

    header = {
        "pwr_up_msec": header_values[0],
        "utc_msec": header_values[1],
        "listening_sec": header_values[2],
        "sos_mps": header_values[3],
        "ping_number": header_values[4],
        "ping_hz": header_values[5],
        "pulse_sec": header_values[6],
        "flags": header_values[7],
        "num_points": header_values[8],
        "reserved": header_values[9],
    }

    point_fmt = "<ffII"
    point_len = struct.calcsize(point_fmt)

    points = []
    offset = header_len

    for _ in range(header["num_points"]):
        angle_rad, tof_s, reserved0, reserved1 = struct.unpack_from(point_fmt, payload, offset)
        offset += point_len

        points.append({
            "angle_rad": angle_rad,
            "tof_s": tof_s,
            "reserved0": reserved0,
            "reserved1": reserved1,
        })

    return header, points


def parse_surveyor_attitude(payload):
    fmt = "<ffffffQI"
    values = struct.unpack_from(fmt, payload, 0)

    return {
        "up_vec_x": values[0],
        "up_vec_y": values[1],
        "up_vec_z": values[2],
        "reserved_x": values[3],
        "reserved_y": values[4],
        "reserved_z": values[5],
        "utc_msec": values[6],
        "pwr_up_msec": values[7],
    }


def utc_iso_from_msec(utc_msec):
    if not utc_msec:
        return ""
    return datetime.fromtimestamp(utc_msec / 1000, tz=timezone.utc).isoformat()


def safe_json_payload(data):
    try:
        text = payload_bytes(data).decode("utf-8", errors="replace")
        return json.loads(text)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Extract Surveyor240 + BlueBoat SVLZ/SVLOG streams.")
    parser.add_argument("input_file")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (defaults to input filename without extension)"
    )
    parser.add_argument("--max_packets", type=int, default=None)
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if args.output_dir is None:
        output_dir = input_path.with_suffix("")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "atof": open(output_dir / "surveyor_atof.csv", "w", newline=""),
        "sonar_att": open(output_dir / "surveyor_attitude.csv", "w", newline=""),
        "bb_pos": open(output_dir / "blueboat_position.csv", "w", newline=""),
        "bb_att": open(output_dir / "blueboat_attitude.csv", "w", newline=""),
        "bb_local": open(output_dir / "blueboat_local_position.csv", "w", newline=""),
        "bb_other": open(output_dir / "blueboat_other_mavlink.jsonl", "w"),
    }

    writers = {
        "atof": csv.writer(files["atof"]),
        "sonar_att": csv.writer(files["sonar_att"]),
        "bb_pos": csv.writer(files["bb_pos"]),
        "bb_att": csv.writer(files["bb_att"]),
        "bb_local": csv.writer(files["bb_local"]),
    }

    writers["atof"].writerow([
        "packet_index", "ping_number", "pwr_up_msec", "utc_msec", "utc_iso",
        "ping_hz", "sos_mps", "listening_sec", "pulse_sec",
        "point_index", "angle_rad", "angle_deg", "tof_s", "range_m", "y_m", "z_m",
        "reserved0", "reserved1",
    ])

    writers["sonar_att"].writerow([
        "packet_index", "pwr_up_msec", "utc_msec", "utc_iso",
        "up_vec_x", "up_vec_y", "up_vec_z",
        "pitch_rad", "roll_rad", "pitch_deg", "roll_deg",
    ])

    writers["bb_pos"].writerow([
        "packet_index", "time_boot_ms",
        "lat_deg", "lon_deg", "alt_m", "relative_alt_m",
        "heading_deg", "vx_mps", "vy_mps", "vz_mps",
    ])

    writers["bb_att"].writerow([
        "packet_index", "time_boot_ms",
        "roll_rad", "pitch_rad", "yaw_rad",
        "roll_deg", "pitch_deg", "yaw_deg",
        "rollspeed_radps", "pitchspeed_radps", "yawspeed_radps",
    ])

    writers["bb_local"].writerow([
        "packet_index", "time_boot_ms",
        "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps",
    ])

    packet_i = 0

    with open_log(args.input_file) as f:
        while True:
            if args.max_packets is not None and packet_i >= args.max_packets:
                break

            data = Surveyor240.read_packet(f)
            if data is None:
                break

            packet_i += 1
            msg_id = data.message_id

            if msg_id == definitions.SURVEYOR240_ATOF_POINT_DATA:
                header, points = parse_atof_payload(payload_bytes(data))
                sos_mps = header["sos_mps"]

                for point_i, p in enumerate(points):
                    angle = p["angle_rad"]
                    tof = p["tof_s"]

                    range_m = 0.5 * sos_mps * tof
                    y_m = range_m * math.sin(angle)
                    z_m = -range_m * math.cos(angle)

                    writers["atof"].writerow([
                        packet_i,
                        header["ping_number"],
                        header["pwr_up_msec"],
                        header["utc_msec"],
                        utc_iso_from_msec(header["utc_msec"]),
                        header["ping_hz"],
                        sos_mps,
                        header["listening_sec"],
                        header["pulse_sec"],
                        point_i,
                        angle,
                        math.degrees(angle),
                        tof,
                        range_m,
                        y_m,
                        z_m,
                        p["reserved0"],
                        p["reserved1"],
                    ])

            elif msg_id == definitions.SURVEYOR240_ATTITUDE_REPORT:
                att = parse_surveyor_attitude(payload_bytes(data))

                pitch = math.asin(att["up_vec_x"])
                roll = math.atan2(att["up_vec_y"], att["up_vec_z"])

                writers["sonar_att"].writerow([
                    packet_i,
                    att["pwr_up_msec"],
                    att["utc_msec"],
                    utc_iso_from_msec(att["utc_msec"]),
                    att["up_vec_x"],
                    att["up_vec_y"],
                    att["up_vec_z"],
                    pitch,
                    roll,
                    math.degrees(pitch),
                    math.degrees(roll),
                ])

            elif msg_id == 150:
                obj = safe_json_payload(data)
                if obj is None:
                    continue

                m = obj.get("message", {})
                msg_type = m.get("type")

                if msg_type == "GLOBAL_POSITION_INT":
                    writers["bb_pos"].writerow([
                        packet_i,
                        m.get("time_boot_ms"),
                        m.get("lat") / 1e7 if m.get("lat") is not None else "",
                        m.get("lon") / 1e7 if m.get("lon") is not None else "",
                        m.get("alt") / 1000 if m.get("alt") is not None else "",
                        m.get("relative_alt") / 1000 if m.get("relative_alt") is not None else "",
                        m.get("hdg") / 100 if m.get("hdg") is not None else "",
                        m.get("vx") / 100 if m.get("vx") is not None else "",
                        m.get("vy") / 100 if m.get("vy") is not None else "",
                        m.get("vz") / 100 if m.get("vz") is not None else "",
                    ])

                elif msg_type == "ATTITUDE":
                    writers["bb_att"].writerow([
                        packet_i,
                        m.get("time_boot_ms"),
                        m.get("roll"),
                        m.get("pitch"),
                        m.get("yaw"),
                        math.degrees(m["roll"]) if m.get("roll") is not None else "",
                        math.degrees(m["pitch"]) if m.get("pitch") is not None else "",
                        math.degrees(m["yaw"]) if m.get("yaw") is not None else "",
                        m.get("rollspeed"),
                        m.get("pitchspeed"),
                        m.get("yawspeed"),
                    ])

                elif msg_type == "LOCAL_POSITION_NED":
                    writers["bb_local"].writerow([
                        packet_i,
                        m.get("time_boot_ms"),
                        m.get("x"),
                        m.get("y"),
                        m.get("z"),
                        m.get("vx"),
                        m.get("vy"),
                        m.get("vz"),
                    ])

                else:
                    files["bb_other"].write(json.dumps({
                        "packet_index": packet_i,
                        "message": obj,
                    }) + "\n")

    for f in files.values():
        f.close()

    print(f"Done. Wrote files to: {output_dir}")


if __name__ == "__main__":
    main()