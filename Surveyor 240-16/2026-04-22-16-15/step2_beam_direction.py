import numpy as np

def beam_direction(angle_deg, yaw_rad, pitch_rad, mount_deg=70.0):
    """
    Converts one sonar beam into a unit direction vector in world frame.

    angle_deg  : beam angle from surveyor_atof (e.g. -34, -28, 0, +20)
    yaw_rad    : boat heading from poses.csv
    pitch_rad  : boat pitch from poses.csv
    mount_deg  : sonar tilt from vertical = 70 degrees (from your diagram)

    Returns: numpy array [dx, dy, dz]
        dx = northing direction  (positive = forward/north)
        dy = easting direction   (positive = east)
        dz = vertical            (negative = pointing down)
    """

    a = np.radians(angle_deg)

    # Beam in sonar frame
    # boresight is along -X, fan sweeps in the XY plane
    b_sonar = np.array([-np.cos(a), np.sin(a), 0.0])

    # Mount rotation: tilt boresight forward and down
    # 70 deg from vertical = 20 deg below horizontal
    mt = np.radians(90.0 - mount_deg)   # = 20 degrees
    cos_m = np.cos(mt)
    sin_m = np.sin(mt)

    b_boat = np.array([
        b_sonar[0] * (-cos_m) + b_sonar[2] * (-sin_m),
        b_sonar[1],
        b_sonar[0] *   sin_m  + b_sonar[2] * (-cos_m)
    ])

    # Apply boat pitch (rotation around Y axis)
    cp = np.cos(pitch_rad)
    sp = np.sin(pitch_rad)
    b_pitched = np.array([
        b_boat[0]*cp + b_boat[2]*sp,
        b_boat[1],
       -b_boat[0]*sp + b_boat[2]*cp
    ])

    # Apply boat yaw / heading (rotation around Z axis)
    cy = np.cos(yaw_rad)
    sy = np.sin(yaw_rad)
    b_world = np.array([
        b_pitched[0]*cy - b_pitched[1]*sy,
        b_pitched[0]*sy + b_pitched[1]*cy,
        b_pitched[2]
    ])

    return b_world


# ── Verify it makes sense ──────────────────────────────────────────
if __name__ == '__main__':

    # Use ping 39791 values from your poses.csv
    yaw   = 0.321758
    pitch = 0.025977

    print("Beam directions for ping 39791:")
    print(f"{'angle':>8}  {'dx(N)':>8}  {'dy(E)':>8}  {'dz(Z)':>8}  {'unit?':>6}")

    for angle in [-34, -28, -23, -16, 0, +16]:
        d = beam_direction(angle, yaw, pitch)
        unit = np.linalg.norm(d)
        print(f"{angle:>8}  {d[0]:>8.4f}  {d[1]:>8.4f}  {d[2]:>8.4f}  {unit:>6.4f}")

    print()
    print("Check:")
    print("  dx > 0  means beam points FORWARD (north)  ← should be true")
    print("  dz < 0  means beam points DOWNWARD         ← should be true")
    print("  unit = 1.0000                              ← should be true")