from setuptools import setup

setup(name='alr_envs',
      version='0.0.1',
      install_requires=['gym',
                        'PyQt5',
                        'matplotlib',
                        'mp_lib @ git+https://git@github.com/maxhuettenrauch/mp_lib@master#egg=mp_lib',
                        'mujoco_py'],  # And any other dependencies foo needs
      )
