import pandas as pd
import numpy as np

# Load the raw data files
atof = pd.read_csv('surveyor_atof.csv')
pos  = pd.read_csv('blueboat_local_position.csv').sort_values('time_boot_ms')
att  = pd.read_csv('blueboat_attitude.csv').sort_values('time_boot_ms')
satt = pd.read_csv('surveyor_attitude.csv').sort_values('pwr_up_msec')

# One timestamp per ping
pings = atof.drop_duplicates('ping_number')[['ping_number','pwr_up_msec']]
pings = pings.sort_values('pwr_up_msec')

# Match each ping to nearest boat position by timestamp
pings = pd.merge_asof(pings,
    pos[['time_boot_ms','x_m','y_m']],
    left_on='pwr_up_msec', right_on='time_boot_ms',
    direction='nearest')

# Match each ping to nearest boat attitude
pings = pd.merge_asof(pings,
    att[['time_boot_ms','yaw_rad','pitch_rad']],
    left_on='pwr_up_msec', right_on='time_boot_ms',
    direction='nearest')

# Rename for clarity
pings = pings.rename(columns={'x_m':'sonar_x', 'y_m':'sonar_y'})
pings['sonar_z'] = 0.0   # sonar sits at water surface

print(f"Total pings: {len(pings)}")
print(pings[['ping_number','sonar_x','sonar_y','yaw_rad','pitch_rad']].head(5))

pings.to_csv('poses.csv', index=False)
print("Saved poses.csv")