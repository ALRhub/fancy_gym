from abc import ABC

from gym import spaces
import numpy as np
from alr_envs.alr.classic_control.base_reacher.base_reacher import BaseReacherEnv


class BaseReacherTorqueEnv(BaseReacherEnv, ABC):
    """
    Base class for torque controlled reaching environments
    """
    def __init__(self, n_links: int, random_start: bool = True,
                 allow_self_collision: bool = False):
        super().__init__(n_links, random_start, allow_self_collision)

        self.max_torque = 1000
        action_bound = np.ones((self.n_links,)) * self.max_torque
        self.action_space = spaces.Box(low=-action_bound, high=action_bound, shape=action_bound.shape)

    def step(self, action: np.ndarray):
        """
        A single step with action in torque space
        """

        self._acc = action
        self._angle_velocity = self._angle_velocity + self.dt * action
        self._joint_angles = self._joint_angles + self.dt * self._angle_velocity
        self._update_joints()

        self._is_collided = self._check_collisions()

        reward, info = self._get_reward(action)

        self._steps += 1
        done = self._terminate(info)

        return self._get_obs().copy(), reward, done, False, info
