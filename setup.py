from setuptools import setup

setup(
    name='alr_envs',
    version='0.0.1',
    packages=['alr_envs', 'alr_envs.classic_control', 'alr_envs.open_ai', 'alr_envs.mujoco', 'alr_envs.dmc',
              'alr_envs.utils'],
    install_requires=[
        'gym',
        'PyQt5',
        'matplotlib',
        'mp_env_api @ git+ssh://git@github.com/ALRhub/motion_primitive_env_api.git',
        'mujoco-py<2.1,>=2.0',
        'dm_control'
    ],

    url='https://github.com/ALRhub/alr_envs/',
    license='MIT',
    author='Fabian Otto, Marcel Sandermann, Maximilian Huettenrauch',
    author_email='',
    description='Custom Gym environments for various (robotics) tasks. integration of DMC environments into the'
                'gym interface, and support for using motion primitives with gym environments.'
)
