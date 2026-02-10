from setuptools import find_packages, setup

package_name = 'gaussian_filters'

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
    maintainer='vboxuser',
    maintainer_email='mmunoria@mtu.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'kf_node = gaussian_filters.kf_node:main',
            'ekf_node = gaussian_filters.ekf_node:main',
            'ukf_node = gaussian_filters.ukf_node:main',
        ],
    },
)
