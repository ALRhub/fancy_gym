from setuptools import setup

setup(
    name='alr_envs',
    version='0.0.1',
    packages=['alr_envs', 'alr_envs.classic_control', 'alr_envs.mujoco', 'alr_envs.stochastic_search',
              'alr_envs.utils'],
    install_requires=[
        'gym',
        'PyQt5',
        'matplotlib',
        'mp_env_api @ git+ssh://git@github.com/ALRhub/motion_primitive_env_api.git',
        'mujoco_py',
        'dm_control'
    ],

    url='https://github.com/ALRhub/alr_envs/',
    license='MIT',
    author='Fabian Otto, Marcel Sandermann, Maximilian Huettenrauch',
    author_email='',
    description='Custom Gym environments for various (robotics) simple_reacher.'
)
