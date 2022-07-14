import itertools

from setuptools import setup, find_packages

# Environment-specific dependencies for dmc and metaworld
extras = {
    "dmc": ["dm_control==1.0.1"],
    "metaworld": ["metaworld @ git+https://github.com/rlworkgroup/metaworld.git@master#egg=metaworld",
                  'mujoco-py<2.2,>=2.1'],
}

# All dependencies
all_groups = set(extras.keys())
extras["all"] = list(set(itertools.chain.from_iterable(map(lambda group: extras[group], all_groups))))

setup(
    author='Fabian Otto, Onur Celik, Marcel Sandermann, Maximilian Huettenrauch',
    name='simple_gym',
    version='0.1',
    classifiers=[
        # Python 3.6 is minimally supported (only with basic gym environments and API)
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    extras_require=extras,
    install_requires=[
        'gym>=0.24.0',
        'mujoco==2.2.0',
    ],
    packages=[package for package in find_packages() if package.startswith("fancy_gym")],
    # packages=['fancy_gym', 'fancy_gym.envs', 'fancy_gym.open_ai', 'fancy_gym.dmc', 'fancy_gym.meta', 'fancy_gym.utils'],
    package_data={
        "fancy_gym": [
            "envs/mujoco/*/assets/*.xml",
        ]
    },
    python_requires=">=3.7",
    url='https://github.com/ALRhub/fancy_gym/',
    # license='AGPL-3.0 license',
    author_email='',
    description='Fancy Gym: Unifying interface for various RL benchmarks with support for Black Box approaches.'
)
