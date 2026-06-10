from setuptools import find_packages, setup

package_name = 'lidar_plotter'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='Live lidar obstacle plotter',
    license='TODO',
    entry_points={
    'console_scripts': [
        'lidar_plotter = lidar_plotter.lidar_plotter:main',
        'lidar_plotter_mod = lidar_plotter.lidar_plotter_mod:main',
        'validate_lidar_against_sdf = lidar_plotter.validate_lidar_against_sdf:main',
        'lidar_truth_overlay = lidar_plotter.lidar_truth_overlay:main',
        ],
    },
)
