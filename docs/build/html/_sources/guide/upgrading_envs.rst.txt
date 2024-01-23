Creating new MP Environments
----------------------------

This guide will explain to you how to upgrade an existing step-based Gymnasium environment into one, that supports Movement Primitives (MPs). If you are looking for a guide to build such a Gymnasium environment instead, please have a look at `this guide <https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/>`__.

In case a required task is not supported yet in the MP framework, it can
be created relatively easy. For the task at hand, the following
`interface <https://github.com/ALRhub/fancy_gym/tree/master/fancy_gym/black_box/raw_interface_wrapper.py>`__
needs to be implemented.

.. code:: python

   from abc import abstractmethod
   from typing import Union, Tuple

   import gymnasium as gym
   import numpy as np


   class RawInterfaceWrapper(gym.Wrapper):
       mp_config = {
           'ProMP': {},
           'DMP': {},
           'ProDMP': {},
       }

       @property
       def context_mask(self) -> np.ndarray:
           """
               Returns boolean mask of the same shape as the observation space.
               It determines whether the observation is returned for the contextual case or not.
               This effectively allows to filter unwanted or unnecessary observations from the full step-based case.
               E.g. Velocities starting at 0 are only changing after the first action. Given we only receive the
               context/part of the first observation, the velocities are not necessary in the observation for the task.
               Returns:
                   bool array representing the indices of the observations
           """
           return np.ones(self.env.observation_space.shape[0], dtype=bool)

       @property
       @abstractmethod
       def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
           """
               Returns the current position of the action/control dimension.
               The dimensionality has to match the action/control dimension.
               This is not required when exclusively using velocity control,
               it should, however, be implemented regardless.
               E.g. The joint positions that are directly or indirectly controlled by the action.
           """
           raise NotImplementedError()

       @property
       @abstractmethod
       def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
           """
               Returns the current velocity of the action/control dimension.
               The dimensionality has to match the action/control dimension.
               This is not required when exclusively using position control,
               it should, however, be implemented regardless.
               E.g. The joint velocities that are directly or indirectly controlled by the action.
           """
           raise NotImplementedError()

Default configurations for MPs can be overitten by defining attributes
in mp_config. Available parameters are documented in the `MP_PyTorch
Userguide <https://github.com/ALRhub/MP_PyTorch/blob/main/doc/README.md>`__.

.. code:: python

   class RawInterfaceWrapper(gym.Wrapper):
       mp_config = {
           'ProMP': {
               'phase_generator_kwargs': {
                   'phase_generator_type': 'linear'
                   # When selecting another generator type, the default configuration will not be merged for the attribute.
               },
               'controller_kwargs': {
                   'p_gains': 0.5 * np.array([1.0, 4.0, 2.0, 4.0, 1.0, 4.0, 1.0]),
                   'd_gains': 0.5 * np.array([0.1, 0.4, 0.2, 0.4, 0.1, 0.4, 0.1]),
               },
               'basis_generator_kwargs': {
                   'num_basis': 3,
                   'num_basis_zero_start': 1,
                   'num_basis_zero_goal': 1,
               },
           },
           'DMP': {},
           'ProDMP': {}.
       }

       [...]

If you created a new task wrapper, feel free to open a PR, so we can
integrate it for others to use as well. Without the integration the task
can still be used. A rough outline can be shown here, for more details
we recommend having a look at the
:ref:`multiple examples <example-mp>`.

If the step-based is already registered with gym, you can simply do the
following:

.. code:: python

   fancy_gym.upgrade(
       id='custom/cool_new_env-v0',
       mp_wrapper=my_custom_MPWrapper
   )

If the step-based is not yet registered with gym we can add both the
step-based and MP-versions via

.. code:: python

   fancy_gym.register(
       id='custom/cool_new_env-v0',
       entry_point=my_custom_env,
       mp_wrapper=my_custom_MPWrapper
   )

From this point on, you can access MP-version of your environments via

.. code:: python

   env = gym.make('custom_ProDMP/cool_new_env-v0')

   rewards = 0
   observation, info = env.reset()

   # number of samples/full trajectories (multiple environment steps)
   for i in range(5):
       ac = env.action_space.sample()
       observation, reward, terminated, truncated, info = env.step(ac)
       rewards += reward

       if terminated or truncated:
           print(rewards)
           rewards = 0
           observation, info = env.reset()