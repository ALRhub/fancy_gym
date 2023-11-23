import numpy as np
from gymnasium import spaces

from fancy_gym.envs.classic_control.base_reacher.base_reacher import BaseReacherEnv


class BaseReacherDirectEnv(BaseReacherEnv):
    """
    Base class for directly controlled reaching environments
    """

    def __init__(self, n_links: int, random_start: bool = True,
                 allow_self_collision: bool = False, **kwargs):
        super().__init__(n_links, random_start, allow_self_collision, **kwargs)

        self.max_vel = 2 * np.pi
        action_bound = np.ones((self.n_links,)) * self.max_vel
        self.action_space = spaces.Box(low=-action_bound, high=action_bound, shape=action_bound.shape)

    def step(self, action: np.ndarray):
        """
        A single step with action in angular velocity space
        """

        self._acc = (action - self._angle_velocity) / self.dt
        self._angle_velocity = action
        self._joint_angles = self._joint_angles + self.dt * self._angle_velocity
        self._update_joints()

        self._is_collided = self._check_collisions()

        reward, info = self._get_reward(action)

        self._steps += 1
        terminated = self._terminate(info)
        truncated = False

        return self._get_obs().copy(), reward, terminated, truncated, info
