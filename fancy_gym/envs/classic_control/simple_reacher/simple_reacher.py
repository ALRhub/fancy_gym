from typing import Iterable, Union, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from gym.core import ObsType

from fancy_gym.envs.classic_control.base_reacher.base_reacher_torque import BaseReacherTorqueEnv


class SimpleReacherEnv(BaseReacherTorqueEnv):
    """
    Simple Reaching Task without any physics simulation.
    Returns no reward until 150 time steps. This allows the agent to explore the space, but requires precise actions
    towards the end of the trajectory.
    """

    def __init__(self, n_links: int, target: Union[None, Iterable] = None, random_start: bool = True,
                 allow_self_collision: bool = False, ):
        super().__init__(n_links, random_start, allow_self_collision)

        # provided initial parameters
        self.inital_target = target

        # temp container for current env state
        self._goal = None

        self._start_pos = np.zeros(self.n_links)

        self.steps_before_reward = 199

        state_bound = np.hstack([
            [np.pi] * self.n_links,  # cos
            [np.pi] * self.n_links,  # sin
            [np.inf] * self.n_links,  # velocity
            [np.inf] * 2,  # x-y coordinates of target distance
            [np.inf]  # env steps, because reward start after n steps TODO: Maybe
        ])
        self.observation_space = spaces.Box(low=-state_bound, high=state_bound, shape=state_bound.shape)

    # @property
    # def start_pos(self):
    #     return self._start_pos

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False,
              options: Optional[dict] = None, ) -> Union[ObsType, Tuple[ObsType, dict]]:
        self._generate_goal()

        return super().reset()

    def _get_reward(self, action: np.ndarray):
        diff = self.end_effector - self._goal
        reward_dist = 0

        if not self.allow_self_collision:
            self._is_collided = self._check_self_collision()

        if self._steps >= self.steps_before_reward:
            reward_dist -= np.linalg.norm(diff)
            # reward_dist = np.exp(-0.1 * diff ** 2).mean()
            # reward_dist = - (diff ** 2).mean()

        reward_ctrl = (action ** 2).sum()
        reward = reward_dist - reward_ctrl
        return reward, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def _terminate(self, info):
        return False

    def _get_obs(self):
        theta = self._joint_angles
        return np.hstack([
            np.cos(theta),
            np.sin(theta),
            self._angle_velocity,
            self.end_effector - self._goal,
            self._steps
        ]).astype(np.float32)

    def _generate_goal(self):

        if self.inital_target is None:

            total_length = np.sum(self.link_lengths)
            goal = np.array([total_length, total_length])
            while np.linalg.norm(goal) >= total_length:
                goal = self.np_random.uniform(low=-total_length, high=total_length, size=2)
        else:
            goal = np.copy(self.inital_target)

        self._goal = goal

    def _check_collisions(self) -> bool:
        return self._check_self_collision()

    def render(self, mode='human'):  # pragma: no cover
        if self.fig is None:
            # Create base figure once on the beginning. Afterwards only update
            plt.ion()
            self.fig = plt.figure()
            ax = self.fig.add_subplot(1, 1, 1)

            # limits
            lim = np.sum(self.link_lengths) + 0.5
            ax.set_xlim([-lim, lim])
            ax.set_ylim([-lim, lim])

            self.line, = ax.plot(self._joints[:, 0], self._joints[:, 1], 'ro-', markerfacecolor='k')
            goal_pos = self._goal.T
            self.goal_point, = ax.plot(goal_pos[0], goal_pos[1], 'gx')
            self.goal_dist, = ax.plot([self.end_effector[0], goal_pos[0]], [self.end_effector[1], goal_pos[1]], 'g--')

            self.fig.show()

        self.fig.gca().set_title(f"Iteration: {self._steps}, distance: {self.end_effector - self._goal}")

        # goal
        goal_pos = self._goal.T
        if self._steps == 1:
            self.goal_point.set_data(goal_pos[0], goal_pos[1])

        # arm
        self.line.set_data(self._joints[:, 0], self._joints[:, 1])

        # distance between end effector and goal
        self.goal_dist.set_data([self.end_effector[0], goal_pos[0]], [self.end_effector[1], goal_pos[1]])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


if __name__ == "__main__":
    env = SimpleReacherEnv(5)
    env.reset()
    for i in range(200):
        ac = env.action_space.sample()
        obs, rew, done, info = env.step(ac)

        env.render()
        if done:
            break
