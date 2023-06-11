from gymnasium.spaces import Box, Dict
from gym.spaces import Box as OldBox
import gymnasium as gym
import numpy as np


class TimeAwareObservation(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """Augment the observation with the current time step in the episode.

    The observation space of the wrapped environment is assumed to be a flat :class:`Box` or flattable :class:`Dict`.
    In particular, pixel observations are not supported. This wrapper will append the current progress within the current episode to the observation.
    The progress will be indicated as a number between 0 and 1.
    """

    def __init__(self, env: gym.Env, enforce_dtype_float32=False):
        """Initialize :class:`TimeAwareObservation` that requires an environment with a flat :class:`Box` or flattable :class:`Dict` observation space.

        Args:
            env: The environment to apply the wrapper
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ObservationWrapper.__init__(self, env)
        allowed_classes = [Box, OldBox, Dict]
        if enforce_dtype_float32:
            assert env.observation_space.dtype == np.float32, 'TimeAwareObservation was given an environment with a dtype!=np.float32 ('+str(
                env.observation_space.dtype)+'). This requirement can be removed by setting enforce_dtype_float32=False.'
        dtype = env.observation_space.dtype

        assert env.observation_space.__class__ in allowed_classes, str(env.observation_space)+' is not supported. Only Box or Dict'

        if env.observation_space.__class__ in [Box, OldBox]:
            low = np.append(env.observation_space.low, 0.0)
            high = np.append(env.observation_space.high, 1.0)

            self.observation_space = Box(low, high, dtype=dtype)
        else:
            import pdb
            pdb.set_trace()
            exit

        self.is_vector_env = getattr(env, "is_vector_env", False)

    def observation(self, observation):
        """Adds to the observation with the current time step.

        Args:
            observation: The observation to add the time step to

        Returns:
            The observation with the time step appended to (relative to total number of steps)
        """
        return np.append(observation, self.t / self.env.spec.max_episode_steps)

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
