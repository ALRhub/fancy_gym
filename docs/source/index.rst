Fancy Gym
=========

Built upon the foundation of
`Gymnasium <https://gymnasium.farama.org/>`__ (a maintained fork of
OpenAI’s renowned Gym library) ``fancy_gym`` offers a comprehensive
collection of reinforcement learning environments.

Key Features
------------

 - **New Challenging Environments**: ``fancy_gym`` includes several new
   environments (Panda Box Pushing, Table Tennis, etc.) that present a
   higher degree of difficulty, pushing the boundaries of reinforcement
   learning research.
 - **Support for Movement Primitives**: ``fancy_gym`` supports a range
   of movement primitives (MPs), including Dynamic Movement Primitives
   (DMPs), Probabilistic Movement Primitives (ProMP), and Probabilistic
   Dynamic Movement Primitives (ProDMP).
 - **Upgrade to Movement Primitives**: With our framework, it’s
   straightforward to transform standard Gymnasium environments into
   environments that support movement primitives.
 - **Benchmark Suite Compatibility**: ``fancy_gym`` makes it easy to
   access renowned benchmark suites such as `DeepMind
   Control <https://deepmind.com/research/publications/2020/dm-control-Software-and-Tasks-for-Continuous-Control>`__
   and `Metaworld <https://meta-world.github.io/>`__, whether you want
   to use them in the regular step-based setting or using MPs.
 - **Contribute Your Own Environments**: If you’re inspired to create
   custom gym environments, both step-based and with movement
   primitives, this
   `guide <https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/>`__
   will assist you. We encourage and highly appreciate submissions via
   PRs to integrate these environments into ``fancy_gym``.

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

To cite this repository in publications:

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

   <div style="text-align: center;">
      <a href="https://alr.anthropomatik.kit.edu/"><img src="_static/imgs/alr.svg" style="margin: 5%; width: 20%;"></a>
      <a href="https://www.kit.edu/"><img src="_static/imgs/kit.svg" style="margin: 5%; width: 20%;"></a>
      <a href="https://uni-tuebingen.de/"><img src="_static/imgs/uni_tuebingen.svg" style="margin: 5%; width: 20%;"></a>
   </div>
