from alr_envs.classic_control.simple_reacher import SimpleReacherEnv
from gym import spaces
import numpy as np


class EpisodicSimpleReacherEnv(SimpleReacherEnv):
    def __init__(self, n_links, random_start=True):
        super(EpisodicSimpleReacherEnv, self).__init__(n_links, random_start)

        # self._goal_pos = None

        if random_start:
            state_bound = np.hstack([
                [np.pi] * self.n_links,  # cos
                [np.pi] * self.n_links,  # sin
                [np.inf] * self.n_links,  # velocity
            ])
        else:
            state_bound = np.empty(0, )

        state_bound = np.hstack([
            state_bound,
            [np.inf] * 2,  # x-y coordinates of goal
        ])

        self.observation_space = spaces.Box(low=-state_bound, high=state_bound, shape=state_bound.shape)

    @property
    def start_pos(self):
        return self._start_pos

    # @property
    # def goal_pos(self):
    #     return self._goal_pos

    def _get_obs(self):
        if self.random_start:
            theta = self._joint_angle
            return np.hstack([
                np.cos(theta),
                np.sin(theta),
                self._angle_velocity,
                self._goal,
            ])
        else:
            return self._goal
