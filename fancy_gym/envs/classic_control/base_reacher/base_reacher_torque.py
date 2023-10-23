import numpy as np
from gymnasium import spaces

from fancy_gym.envs.classic_control.base_reacher.base_reacher import BaseReacherEnv


class BaseReacherTorqueEnv(BaseReacherEnv):
    """
    Base class for torque controlled reaching environments
    """

    def __init__(self, n_links: int, random_start: bool = True,
                 allow_self_collision: bool = False, **kwargs):
        super().__init__(n_links, random_start, allow_self_collision, **kwargs)

        self.max_torque = 1000
        action_bound = np.ones((self.n_links,)) * self.max_torque
        self.action_space = spaces.Box(low=-action_bound, high=action_bound, shape=action_bound.shape)

    def step(self, action: np.ndarray):
        """
        A single step with action in torque space
        """

        self._angle_velocity = self._angle_velocity + self.dt * action
        self._joint_angles = self._joint_angles + self.dt * self._angle_velocity
        self._update_joints()

        self._is_collided = self._check_collisions()

        reward, info = self._get_reward(action)

        self._steps += 1
        terminated = False
        truncated = False

        return self._get_obs().copy(), reward, terminated, truncated, info
