"""
Adapted from: https://github.com/openai/gym/blob/907b1b20dd9ac0cba5803225059b9c6673702467/gym/wrappers/time_aware_observation.py
License: MIT
Copyright (c) 2016 OpenAI (https://openai.com)

Wrapper for adding time aware observations to environment observation.
"""
import gym
import numpy as np
from gym.spaces import Box


class TimeAwareObservation(gym.ObservationWrapper):
    """Augment the observation with the current time step in the episode.

    The observation space of the wrapped environment is assumed to be a flat :class:`Box`.
    In particular, pixel observations are not supported. This wrapper will append the current timestep
     within the current episode to the observation.

    Example:
        >>> import gym
        >>> env = gym.make('CartPole-v1')
        >>> env = TimeAwareObservation(env)
        >>> env.reset()
        array([ 0.03810719,  0.03522411,  0.02231044, -0.01088205,  0.        ])
        >>> env.step(env.action_space.sample())[0]
        array([ 0.03881167, -0.16021058,  0.0220928 ,  0.28875574,  1.        ])
    """

    def __init__(self, env: gym.Env):
        """Initialize :class:`TimeAwareObservation` that requires an environment with a flat :class:`Box`
        observation space.

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        assert isinstance(env.observation_space, Box)
        low = np.append(self.observation_space.low, 0.0)
        high = np.append(self.observation_space.high, 1.0)
        self.observation_space = Box(low, high, dtype=self.observation_space.dtype)
        self.t = 0
        self._max_episode_steps = env.spec.max_episode_steps

    def observation(self, observation):
        """Adds to the observation with the current time step normalized with max steps.

        Args:
            observation: The observation to add the time step to

        Returns:
            The observation with the time step appended to
        """
        return np.append(observation, self.t / self._max_episode_steps)

    def step(self, action):
        """Steps through the environment, incrementing the time step.

        Args:
            action: The action to take

        Returns:
            The environment's step using the action.
        """
        self.t += 1
        return super().step(action)

    def reset(self, **kwargs):
        """Reset the environment setting the time to zero.

        Args:
            **kwargs: Kwargs to apply to env.reset()

        Returns:
            The reset environment
        """
        self.t = 0
        return super().reset(**kwargs)
