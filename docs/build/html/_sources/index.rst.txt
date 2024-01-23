Fancy Gym
=========

.. raw:: html

   <div style="text-align: center;">
      <img src="_static/imgs/fancy_namelogo.svg" style="margin: 5%; width: 80%;"></a>
   </div>
    <style>
       /* Little Hack: We don't want to show the title (ugly), but need to define it since it also sets the pages metadata (for titlebar and stuff) */
        h1 {
            display: none;
        }
    </style>


Built upon the foundation of
`Gymnasium <https://gymnasium.farama.org/>`__ (a maintained fork of
OpenAI’s renowned Gym library) ``fancy_gym`` offers a comprehensive
collection of reinforcement learning environments.

Key Features
------------

 - **New Challenging Environments**: ``fancy_gym`` includes several new
   environments (`Panda Box Pushing <envs/fancy/mujoco.html#box-pushing>`_,
   `Table Tennis <envs/fancy/mujoco.html#table-tennis>`_,
   `etc. <envs/fancy/index.html>`_) that present a higher degree of
   difficulty, pushing the boundaries of reinforcement learning research.
 - **Support for Movement Primitives**: ``fancy_gym`` supports a range
   of movement primitives (MPs), including Dynamic Movement Primitives
   (DMPs), Probabilistic Movement Primitives (ProMP), and Probabilistic
   Dynamic Movement Primitives (ProDMP).
 - **Upgrade to Movement Primitives**: With our framework, it’s
   straightforward to transform standard Gymnasium environments into
   environments that support movement primitives.
 - **Benchmark Suite Compatibility**: ``fancy_gym`` makes it easy to
   access renowned benchmark suites such as `DeepMind
   Control <envs/dmc.html>`__
   and `Metaworld <envs/meta.html>`__, whether you want
   to use them in the regular step-based setting or using MPs.
 - **Contribute Your Own Environments**: If you’re inspired to create
   custom gym environments, both step-based and with movement
   primitives, this
   `guide <guide/upgrading_envs.html>`__
   will assist you. We encourage and highly appreciate submissions via
   PRs to integrate these environments into ``fancy_gym``.

Quickstart Guide
----------------

Install via pip (`or use an alternative installation method <guide/installation.html>`__)

.. code:: bash

   pip install 'fancy_gym[all]'


Try out one of our step-based environments (`or explore our other envs <envs/fancy/index.html>`__)

.. code:: python

   import gymnasium as gym
   import fancy_gym
   import time

   env = gym.make('fancy/BoxPushingDense-v0', render_mode='human')
   observation = env.reset()
   env.render()

   for i in range(1000):
      action = env.action_space.sample() # Randomly sample an action
      observation, reward, terminated, truncated, info = env.step(action)
      time.sleep(1/env.metadata['render_fps'])

      if terminated or truncated:
            observation, info = env.reset()


Explore the MP-based variant (`or learn more about Movement Primitives (MPs) <guide/episodic_rl.html>`__)

.. code:: python

   import gymnasium as gym
   import fancy_gym

   env = gym.make('fancy_ProMP/BoxPushingDense-v0', render_mode='human')
   env.reset()
   env.render()
   
   for i in range(10):
      action = env.action_space.sample() # Randomly sample MP parameters
      observation, reward, terminated, truncated, info = env.step(action) # Will execute full trajectory, based on MP
      observation = env.reset()


.. toctree::
   :maxdepth: 3
   :caption: User Guide

   guide/installation
   guide/episodic_rl
   guide/basic_usage
   guide/upgrading_envs

.. toctree::
   :maxdepth: 3
   :caption: Environments

   envs/fancy/index
   envs/dmc
   envs/meta
   envs/open_ai

.. toctree::
   :maxdepth: 3
   :caption: Examples

   examples/general
   examples/dmc
   examples/metaworld
   examples/open_ai
   examples/movement_primitives
   examples/mp_params_tuning
   examples/pd_control_gain_tuning
   examples/replanning_envs

.. toctree::
   :maxdepth: 3
   :caption: API

   api

Citing the Project
------------------

To cite `fancy_gym` in publications:

.. code:: bibtex

   @software{fancy_gym,
       title = {Fancy Gym},
       author = {Otto, Fabian and Celik, Onur and Roth, Dominik and Zhou, Hongyi},
       abstract = {Fancy Gym: Unifying interface for various RL benchmarks with support for Black Box approaches.},
       url = {https://github.com/ALRhub/fancy_gym},
       organization = {Autonomous Learning Robots Lab (ALR) at KIT},
   }

Icon Attribution
----------------

The icon is based on the
`Gymnasium <https://github.com/Farama-Foundation/Gymnasium>`__ icon as
can be found
`here <https://gymnasium.farama.org/_static/img/gymnasium_black.svg>`__.

=================

.. raw:: html

   <div style="text-align: center; background: #f8f8f8; border-radius: 10px;">
      <a href="https://alr.iar.kit.edu/"><img src="_static/imgs/alr.svg" style="margin: 5%; width: 20%;"></a>
      <a href="https://www.kit.edu/"><img src="_static/imgs/kit.svg" style="margin: 5%; width: 20%;"></a>
      <a href="https://uni-tuebingen.de/"><img src="_static/imgs/uni_tuebingen.svg" style="margin: 5%; width: 20%;"></a>
   </div>
   <br>
