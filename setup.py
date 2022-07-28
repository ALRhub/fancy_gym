import itertools

from setuptools import setup, find_packages

# Environment-specific dependencies for dmc and metaworld
extras = {
    "dmc": ["dm_control>=1.0.1"],
    "metaworld": ["metaworld @ git+https://github.com/rlworkgroup/metaworld.git@master#egg=metaworld",
                  'mujoco-py<2.2,>=2.1'],
}

# All dependencies
all_groups = set(extras.keys())
extras["all"] = list(set(itertools.chain.from_iterable(map(lambda group: extras[group], all_groups))))

setup(
    author='Fabian Otto, Onur Celik',
    name='fancy_gym',
    version='0.2',
    classifiers=[
        # Python 3.7 is minimally supported
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    extras_require=extras,
    install_requires=[
        'gym[mujoco]<0.25.0,>=0.24.0',
        'mp_pytorch @ git+https://github.com/ALRhub/MP_PyTorch.git@main'
    ],
    packages=[package for package in find_packages() if package.startswith("fancy_gym")],
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
