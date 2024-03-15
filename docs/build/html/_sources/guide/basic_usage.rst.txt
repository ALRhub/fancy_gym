Basic Usage
-----------

We will only show the basics here and prepared :ref:`multiple examples <example-general>` for a more detailed look.

Step-Based Environments
~~~~~~~~~~~~~~~~~~~~~~~

Regular step based environments added by Fancy Gym are added into the
``fancy/`` namespace.

.. note::
    Legacy versions of Fancy Gym used ``fancy_gym.make(...)``. This is no longer supported and will raise an Exception on new versions.

.. code:: python

   import gymnasium as gym
   import fancy_gym

   env = gym.make('fancy/Reacher5d-v0')
   # or env = gym.make('metaworld/reach-v2') # fancy_gym allows access to all metaworld ML1 tasks via the metaworld/ NS
   # or env = gym.make('dm_control/ball_in_cup-catch-v0')
   # or env = gym.make('Reacher-v2')
   observation = env.reset(seed=1)

   for i in range(1000):
       action = env.action_space.sample()
       observation, reward, terminated, truncated, info = env.step(action)
       if i % 5 == 0:
           env.render()

       if terminated or truncated:
           observation, info = env.reset()

Black-Box Environments
~~~~~~~~~~~~~~~~~~~~~~

All environments provide by default the cumulative episode reward, this
can however be changed if necessary. Optionally, each environment
returns all collected information from each step as part of the infos.
This information is, however, mainly meant for debugging as well as
logging and not for training.

+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------+----------+
| Key                 | Description                                                                                                                                | Type     |
+=====================+============================================================================================================================================+==========+
| `positions`         | Generated trajectory from MP                                                                                                               | Optional |
+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------+----------+
| `velocities`        | Generated trajectory from MP                                                                                                               | Optional |
+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------+----------+
| `step_actions`      | Step-wise executed action based on controller output                                                                                       | Optional |
+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------+----------+
| `step_observations` | Step-wise intermediate observations                                                                                                        | Optional |
+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------+----------+
| `step_rewards`      | Step-wise rewards                                                                                                                          | Optional |
+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------+----------+
| `trajectory_length` | Total number of environment interactions                                                                                                   | Always   |
+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------+----------+
| `other`             | All other information from the underlying environment are returned as a list with length `trajectory_length` maintaining the original key. | Always   |
|                     | In case some information are not provided every time step, the missing values are filled with `None`.                                      |          |
+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------+----------+

Existing MP tasks can be created the same way as above. The namespace of
a MP-variant of an environment is given by
``<original namespace>_<MP name>/``. Just keep in mind, calling
``step()`` executes a full trajectory.

.. note::
    Currently, we are also in the process of enabling replanning as
    well as learning of sub-trajectories. This allows to split the
    episode into multiple trajectories and is a hybrid setting between
    step-based and black-box leaning. While this is already
    implemented, it is still in beta and requires further testing. Feel
    free to try it and open an issue with any problems that occur.

.. code:: python

   import gymnasium as gym
   import fancy_gym

   env = gym.make('fancy_ProMP/Reacher5d-v0')
   # or env = gym.make('metaworld_ProDMP/reach-v2')
   # or env = gym.make('dm_control_DMP/ball_in_cup-catch-v0')
   # or env = gym.make('gym_ProMP/Reacher-v2') # mp versions of envs added directly by gymnasium are in the gym_<MP-type> NS

   # render() can be called once in the beginning with all necessary arguments.
   # To turn it of again just call render() without any arguments.
   env.render(mode='human')

   # This returns the context information, not the full state observation
   observation, info = env.reset(seed=1)

   for i in range(5):
       action = env.action_space.sample()
       observation, reward, terminated, truncated, info = env.step(action)

       # terminated or truncated is always True as we are working on the episode level, hence we always reset()
       observation, info = env.reset()

To show all available environments, we provide some additional
convenience variables. All of them return a dictionary with the keys
``DMP``, ``ProMP``, ``ProDMP`` and ``all`` that store a list of
available environment ids.

.. code:: python

   import fancy_gym

   print("All Black-box tasks:")
   print(fancy_gym.ALL_MOVEMENT_PRIMITIVE_ENVIRONMENTS)

   print("Fancy Black-box tasks:")
   print(fancy_gym.ALL_FANCY_MOVEMENT_PRIMITIVE_ENVIRONMENTS)

   print("OpenAI Gym Black-box tasks:")
   print(fancy_gym.ALL_GYM_MOVEMENT_PRIMITIVE_ENVIRONMENTS)

   print("Deepmind Control Black-box tasks:")
   print(fancy_gym.ALL_DMC_MOVEMENT_PRIMITIVE_ENVIRONMENTS)

   print("MetaWorld Black-box tasks:")
   print(fancy_gym.ALL_METAWORLD_MOVEMENT_PRIMITIVE_ENVIRONMENTS)

   print("If you add custom envs, their mp versions will be found in:")
   print(fancy_gym.MOVEMENT_PRIMITIVE_ENVIRONMENTS_FOR_NS['<my_custom_namespace>'])