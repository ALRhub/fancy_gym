import itertools
import pathlib

from setuptools import setup, find_packages

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

# Environment-specific dependencies for dmc and metaworld
extras = {
    "dmc": ["dm_control>=1.0.1"],
    "metaworld": ["metaworld @ git+https://github.com/rlworkgroup/metaworld.git@master#egg=metaworld",
                  'mujoco-py<2.2,>=2.1',
                  'scipy'
                  ],
    "test": ["pytest"]
}

# All dependencies
all_groups = set(extras.keys())
extras["all"] = list(set(itertools.chain.from_iterable(map(lambda group: extras[group], all_groups))))

setup(
    name='fancy_gym',
    version='0.3.0',
    description='Fancy Gym: Unifying interface for various RL benchmarks with support for Black Box approaches.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/ALRhub/fancy_gym/',
    author='Fabian Otto, Onur Celik',
    author_email='',
    classifiers=[
        'Development Status :: 3 - Alpha',
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
    ],
    keywords="reinforcement learning, robotics, OpenAI gym, machine learning",
    packages=[package for package in find_packages() if package.startswith("fancy_gym")],
    python_requires=">=3.7, <4",
    install_requires=[
        'gym[mujoco]<0.25.0,>=0.24.0',
        'mp_pytorch @ git+https://github.com/ALRhub/MP_PyTorch.git@main'
    ],
    extras_require=extras,
    package_data={
        "fancy_gym": [
            "envs/mujoco/*/assets/*.xml",
        ]
    },
)
