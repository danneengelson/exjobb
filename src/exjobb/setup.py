from setuptools import setup
import os
from glob import glob
package_name = 'exjobb'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'main = exjobb.main:main',
            'exp1 = exjobb.Experiment1:main',
            'exp2 = exjobb.Experiment2:main',
            'exp3 = exjobb.ExperimentParameter:main',
            'exp4 = exjobb.Experiment3:main',
            'fig2 = exjobb.Experiment3Show:main',
            'fig3 = exjobb.Experiment4Show:main',
            'fig = exjobb.ExperimentShowResults:main',
            'opt = exjobb.Hyperopt:main',
            'show_latest = exjobb.show_latest:main',
        ],
    },
)
