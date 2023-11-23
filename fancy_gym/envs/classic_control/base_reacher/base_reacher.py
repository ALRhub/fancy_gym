from typing import Union, Tuple, Optional, Any, Dict

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ObsType
from gymnasium.utils import seeding

from fancy_gym.envs.classic_control.utils import intersect


class BaseReacherEnv(gym.Env):
    """
    Base class for all reaching environments.
    """

    def __init__(self, n_links: int, random_start: bool = True, allow_self_collision: bool = False, render_mode: str = None):
        super().__init__()
        self.link_lengths = np.ones(n_links)
        self.n_links = n_links
        self._dt = 0.01

        self.render_mode = render_mode

        self.random_start = random_start

        self.allow_self_collision = allow_self_collision

        # state
        self._joints = None
        self._joint_angles = None
        self._angle_velocity = None
        self._acc = None
        self._start_pos = np.hstack([[np.pi / 2], np.zeros(self.n_links - 1)])
        self._start_vel = np.zeros(self.n_links)

        # joint limits
        self.j_min = -np.pi * np.ones(n_links)
        self.j_max = np.pi * np.ones(n_links)

        self.steps_before_reward = 199

        state_bound = np.hstack([
            [np.pi] * self.n_links,  # cos
            [np.pi] * self.n_links,  # sin
            [np.inf] * self.n_links,  # velocity
            [np.inf] * 2,  # x-y coordinates of target distance
            [np.inf]  # env steps, because reward start after n steps TODO: Maybe
        ])

        self.observation_space = spaces.Box(low=-state_bound, high=state_bound, shape=state_bound.shape)

        self.reward_function = None  # Needs to be set in sub class

        # containers for plotting
        self.metadata = {'render.modes': ["human"]}
        self.fig = None

        self._steps = 0

    @property
    def dt(self) -> Union[float, int]:
        return self._dt

    @property
    def current_pos(self):
        return self._joint_angles.copy()

    @property
    def current_vel(self):
        return self._angle_velocity.copy()

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) \
            -> Tuple[ObsType, Dict[str, Any]]:
        # Sample only orientation of first link, i.e. the arm is always straight.
        super(BaseReacherEnv, self).reset(seed=seed, options=options)
        try:
            random_start = options.get('random_start', self.random_start)
        except AttributeError:
            random_start = self.random_start
        if random_start:
            first_joint = self.np_random.uniform(np.pi / 4, 3 * np.pi / 4)
            self._joint_angles = np.hstack([[first_joint], np.zeros(self.n_links - 1)])
            self._start_pos = self._joint_angles.copy()
        else:
            self._joint_angles = self._start_pos

        self._angle_velocity = self._start_vel
        self._joints = np.zeros((self.n_links + 1, 2))
        self._update_joints()
        self._steps = 0

        return self._get_obs().copy(), {}

    def _update_joints(self):
        """
        update joints to get new end-effector position. The other links are only required for rendering.
        Returns:

        """
        angles = np.cumsum(self._joint_angles)
        x = self.link_lengths * np.vstack([np.cos(angles), np.sin(angles)])
        self._joints[1:] = self._joints[0] + np.cumsum(x.T, axis=0)

    def _check_self_collision(self):
        """Checks whether line segments intersect"""

        if self.allow_self_collision:
            return False

        if np.any(self._joint_angles > self.j_max) or np.any(self._joint_angles < self.j_min):
            return True

        link_lines = np.stack((self._joints[:-1, :], self._joints[1:, :]), axis=1)
        for i, line1 in enumerate(link_lines):
            for line2 in link_lines[i + 2:, :]:
                if intersect(line1[0], line1[-1], line2[0], line2[-1]):
                    return True
        return False

    def _get_reward(self, action: np.ndarray) -> (float, dict):
        raise NotImplementedError

    def _get_obs(self) -> np.ndarray:
        raise NotImplementedError

    def _check_collisions(self) -> bool:
        raise NotImplementedError

    def _terminate(self, info) -> bool:
        raise NotImplementedError

    def close(self):
        super(BaseReacherEnv, self).close()
        del self.fig

    @property
    def end_effector(self):
        return self._joints[self.n_links].T
