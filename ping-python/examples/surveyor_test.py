from brping import Surveyor240, definitions
import math
import sys

SONAR_IP = "192.168.2.86"   # change this
SONAR_PORT = 62312         # change only if needed

sonar = Surveyor240()
sonar.connect_udp(SONAR_IP, SONAR_PORT)

if not sonar.initialize():
    print("Failed to initialize Surveyor240")
    sys.exit(1)

print("Connected to Surveyor240")

sonar.control_set_ping_parameters(
    ping_enable=True,
    enable_yz_point_data=True,
    enable_atof_data=True,
)

print("Pinging... Press Ctrl+C to stop")

try:
    while True:
        data = sonar.wait_message([
            definitions.SURVEYOR240_ATOF_POINT_DATA,
            definitions.SURVEYOR240_ATTITUDE_REPORT,
            definitions.SURVEYOR240_YZ_POINT_DATA,
            definitions.SURVEYOR240_WATER_STATS
        ])

        if not data:
            continue

        if data.message_id == definitions.SURVEYOR240_ATTITUDE_REPORT:
            vector = (data.up_vec_x, data.up_vec_y, data.up_vec_z)
            pitch = math.asin(vector[0])
            roll = math.atan2(vector[1], vector[2])
            print(f"ATTITUDE pitch={pitch:.4f}, roll={roll:.4f}")

        elif data.message_id == definitions.SURVEYOR240_ATOF_POINT_DATA:
            print("Received ATOF packet")
            try:
                points = Surveyor240.create_atof_list(data)
                print(f"ATOF points: {len(points)}")
                print(points[:5])
            except Exception as e:
                print(f"Failed to decode ATOF: {e}")

        elif data.message_id == definitions.SURVEYOR240_YZ_POINT_DATA:
            print("Received YZ packet")
            try:
                points = Surveyor240.create_yz_point_data(data)
                print(points[:5])
            except Exception as e:
                print(f"Failed to decode YZ: {e}")

        elif data.message_id == definitions.SURVEYOR240_WATER_STATS:
            print("Received WATER_STATS packet")

except KeyboardInterrupt:
    print("\nStopping pinging...")
    sonar.control_set_ping_parameters(ping_enable=False)
    if sonar.iodev:
        sonar.iodev.close()
