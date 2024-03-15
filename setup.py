# We still provide a setup.py for backwards compatability.
# But the pyproject.toml should be prefered.
import toml
import itertools
from pathlib import Path
from typing import List

from setuptools import setup, find_packages

print('[!] You are currently installing/building fancy_gym via setup.py. This is only provided for backwards-compatability. Please use the pyproject.toml instead.')

pyproject_content = toml.load("pyproject.toml")
project_version = pyproject_content["project"]["version"]

# Environment-specific dependencies for dmc and metaworld
extras = {
    'dmc': ['shimmy[dm-control]', 'Shimmy==1.0.0'],
    'box2d': ['gymnasium[box2d]>=0.26.0'],
    'mujoco-legacy': ['mujoco-py >=2.1,<2.2', 'cython<3'],
    'jax': ["jax >=0.4.0", "jaxlib >=0.4.0"],
    'mushroom-rl': ['mushroom-rl'],
}

# All dependencies
all_groups = set(extras.keys())
extras["all"] = list(set(itertools.chain.from_iterable(
    map(lambda group: extras[group], all_groups))))

extras['testing'] = extras["all"] + ['pytest']


def find_package_data(extensions_to_include: List[str]) -> List[str]:
    envs_dir = Path("fancy_gym/envs/mujoco")
    package_data_paths = []
    for extension in extensions_to_include:
        package_data_paths.extend([str(path)[10:]
                                  for path in envs_dir.rglob(extension)])

    return package_data_paths


setup(
    author='Fabian Otto, Onur Celik, Dominik Roth, Hongyi Zhou',
    name='fancy_gym',
    version=project_version,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    extras_require=extras,
    install_requires=[
        'toml',
        'mp_pytorch<=0.1.3',
        'mujoco==2.3.3',
        'gymnasium[mujoco]>=0.26.0'
    ],
    packages=[package for package in find_packages(
    ) if package.startswith("fancy_gym")],
    package_data={
        "fancy_gym": find_package_data(extensions_to_include=["*.stl", "*.xml"])
    },
    python_requires=">=3.7",
    url='https://github.com/ALRhub/fancy_gym/',
    license='MIT license',
    author_email='',
    description='Fancy Gym: Unifying interface for various RL benchmarks with support for Black Box approaches.'
)
