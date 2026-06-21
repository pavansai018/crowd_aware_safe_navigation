from setuptools import setup
import os
from glob import glob

package_name = 'dr_spaam_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    # package_dir={'': 'src'},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'model_weight'), glob('model_weight/*')),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='pavansai018',
    maintainer_email='18pavansai@gmail.com',
    description='Package for detecting pedestrians using DROW3 or DR-SPAAM in ROS 2.',
    license='BSD-3-Clause',
    extras_require={
        "test": ["pytest"],
    },
    entry_points={
        'console_scripts': [
            'dr_spaam_w_score_ros = dr_spaam_ros2.dr_spaam_w_score_ros:main',
        ],
    },
)
