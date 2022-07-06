import itertools

from setuptools import setup

# Environment-specific dependencies for dmc and metaworld
extras = {
    "dmc": ["dm_control"],
    "meta": ["mujoco_py<2.2,>=2.1, git+https://github.com/rlworkgroup/metaworld.git@master#egg=metaworld"],
    "mujoco": ["mujoco==2.2.0", "imageio>=2.14.1"],
}

# All dependencies
all_groups = set(extras.keys())
extras["all"] = list(set(itertools.chain.from_iterable(map(lambda group: extras[group], all_groups))))

setup(
    author='Fabian Otto, Onur Celik, Marcel Sandermann, Maximilian Huettenrauch',
    name='simple_gym',
    version='0.0.1',
    packages=['alr_envs', 'alr_envs.alr', 'alr_envs.open_ai', 'alr_envs.dmc', 'alr_envs.meta', 'alr_envs.utils'],
    install_requires=[
        'gym',
        'PyQt5',
        # 'matplotlib',
        # 'mp_env_api @ git+https://github.com/ALRhub/motion_primitive_env_api.git',
        #         'mp_env_api @ git+ssh://git@github.com/ALRhub/motion_primitive_env_api.git',
        'mujoco-py<2.1,>=2.0',
        'dm_control',
        'metaworld @ git+https://github.com/rlworkgroup/metaworld.git@master#egg=metaworld',
    ],
    url='https://github.com/ALRhub/alr_envs/',
    # license='AGPL-3.0 license',
    author_email='',
    description='Simple Gym: Aggregating interface for various RL environments with support for Black Box approaches.'
)
