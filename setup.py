import itertools

from setuptools import setup, find_packages

# Environment-specific dependencies for dmc and metaworld
extras = {
    "dmc": ["dm_control==1.0.1"],
    "meta": ["metaworld @ git+https://github.com/rlworkgroup/metaworld.git@master#egg=metaworld"],
    "mujoco": ["mujoco==2.2.0", "imageio>=2.14.1"],
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
        "mujoco_py<2.2,>=2.1",
    ],
    packages=[package for package in find_packages() if package.startswith("alr_envs")],
    # packages=['alr_envs', 'alr_envs.alr', 'alr_envs.open_ai', 'alr_envs.dmc', 'alr_envs.meta', 'alr_envs.utils'],
    package_data={
        "alr_envs": [
            "alr/mujoco/*/assets/*.xml",
        ]
    },
    python_requires=">=3.6",
    url='https://github.com/ALRhub/alr_envs/',
    # license='AGPL-3.0 license',
    author_email='',
    description='Simple Gym: Aggregating interface for various RL environments with support for Black Box approaches.'
)
