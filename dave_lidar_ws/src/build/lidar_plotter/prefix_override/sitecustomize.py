import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/safayat/dave_lidar_ws/src/install/lidar_plotter'
