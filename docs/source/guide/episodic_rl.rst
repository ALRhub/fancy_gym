What is Episodic RL?
--------------------

.. raw:: html

    <div class="justify">

Movement primitive (MP) environments differ from traditional step-based
environments. They align more with concepts from stochastic search,
black-box optimization, and methods commonly found in classical robotics
and control. Instead of individual steps, MP environments operate on an
episode basis, executing complete trajectories. These trajectories are
produced by trajectory generators like Dynamic Movement Primitives
(DMP), Probabilistic Movement Primitives (ProMP) or Probabilistic
Dynamic Movement Primitives (ProDMP).


Once generated, these trajectories are converted into step-by-step
actions using a trajectory tracking controller. The specific controller
chosen depends on the environment’s requirements. Currently, we support
position, velocity, and PD-Controllers tailored for position, velocity,
and torque control. Additionally, we have a specialized controller
designed for the MetaWorld control suite.


While the overarching objective of MP environments remains the learning
of an optimal policy, the actions here represent the parametrization of
motion primitives to craft the right trajectory. Our framework further
enhances this by accommodating a contextual setting. At the episode’s
onset, we present the context space—a subset of the observation space.
This demands the prediction of a new action or MP parametrization for
every unique context.

.. raw:: html

    </div>
