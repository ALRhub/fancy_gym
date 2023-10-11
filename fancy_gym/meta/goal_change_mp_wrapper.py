import numpy as np

from fancy_gym.meta.base_metaworld_mp_wrapper import BaseMetaworldMPWrapper


class MPWrapper(BaseMetaworldMPWrapper):
    """
    This Wrapper is for environments where merely the goal changes in the beginning
    and no secondary objects or end effectors are altered at the start of an episode.
    You can verify this by executing the code below for your environment id and check if the output is non-zero
    at the same indices.
    """

    @property
    def context_mask(self) -> np.ndarray:
        # This structure is the same for all metaworld environments.
        # Only the observations which change could differ
        return np.hstack([
            # Current observation
            [False] * 3,  # end-effector position
            [False] * 1,  # normalized gripper open distance
            [False] * 3,  # main object position
            [False] * 4,  # main object quaternion
            [False] * 3,  # secondary object position
            [False] * 4,  # secondary object quaternion
            # Previous observation
            # TODO: Include previous values? According to their source they might be wrong for the first iteration.
            [False] * 3,  # previous end-effector position
            [False] * 1,  # previous normalized gripper open distance
            [False] * 3,  # previous main object position
            [False] * 4,  # previous main object quaternion
            [False] * 3,  # previous second object position
            [False] * 4,  # previous second object quaternion
            # Goal
            [True] * 3,  # goal position
        ])
