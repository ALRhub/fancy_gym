import gym
import numpy as np
from gym import spaces, utils
from gym.utils import seeding

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('Qt5Agg')  # or can use 'TkAgg', whatever you have/prefer


class SimpleReacherEnv(gym.Env, utils.EzPickle):
    """
    Simple Reaching Task without any physics simulation.
    Returns no reward until 150 time steps. This allows the agent to explore the space, but requires precise actions
    towards the end of the trajectory.
    """

    def __init__(self, n_links):
        super().__init__()
        self.link_lengths = np.ones(n_links)
        self.n_links = n_links
        self.dt = 0.1

        self._goal_pos = None

        self.joints = None
        self._joint_angle = None
        self._angle_velocity = None

        self.max_torque = 1  # 10

        action_bound = np.ones((self.n_links,))
        state_bound = np.hstack([
            [np.pi] * self.n_links,
            [np.inf] * self.n_links,
            [np.inf],
            [np.inf]  # TODO: Maybe
        ])
        self.action_space = spaces.Box(low=-action_bound, high=action_bound, shape=action_bound.shape)
        self.observation_space = spaces.Box(low=-state_bound, high=state_bound, shape=state_bound.shape)

        self.fig = None
        self.metadata = {'render.modes': ["human"]}

        self._steps = 0
        self.seed()

    def step(self, action):

        action = self._scale_action(action)

        self._angle_velocity = self._angle_velocity + self.dt * action
        self._joint_angle = angle_normalize(self._joint_angle + self.dt * self._angle_velocity)
        self._update_joints()
        self._steps += 1

        reward = self._get_reward(action)

        # done = np.abs(self.end_effector - self._goal_pos) < 0.1
        done = False

        return self._get_obs().copy(), reward, done, {}

    def _scale_action(self, action):
        """
        scale actions back in order to provide normalized actions \in [0,1]

        Args:
            action: action to scale

        Returns: action according to self.max_torque

        """

        ub = self.max_torque
        lb = -self.max_torque

        action = lb + (action + 1.) * 0.5 * (ub - lb)
        return np.clip(action, lb, ub)

    def _get_obs(self):
        return [self._joint_angle, self._angle_velocity, self.end_effector - self._goal_pos, self._steps]

    def _update_joints(self):
        """
        update joints to get new end effector position. The other links are only required for rendering.
        Returns:

        """
        angles = np.cumsum(self._joint_angle)
        x = self.link_lengths * np.vstack([np.cos(angles), np.sin(angles)])
        self.joints[1:] = self.joints[0] + np.cumsum(x.T, axis=0)

    def _get_reward(self, action):
        diff = self.end_effector - self._goal_pos
        distance = 0

        # TODO: Is this the best option
        if self._steps > 150:
            distance = np.exp(-0.1 * diff ** 2).mean()
            # distance -= (diff ** 2).mean()

        # distance -= action ** 2
        return distance

    def reset(self):

        # TODO: maybe do initialisation more random?
        # Sample only orientation of first link, i.e. the arm is always straight.
        self._joint_angle = np.hstack([[self.np_random.uniform(-np.pi, np.pi)], np.zeros(self.n_links - 1)])
        self._angle_velocity = np.zeros(self.n_links)
        self.joints = np.zeros((self.n_links + 1, 2))
        self._update_joints()

        self._goal_pos = self._get_random_goal()
        return self._get_obs().copy()

    def _get_random_goal(self):
        center = self.joints[0]

        # Sample uniformly in circle with radius R around center of reacher.
        R = np.sum(self.link_lengths)
        r = R * np.sqrt(self.np_random.uniform())
        theta = self.np_random.uniform() * 2 * np.pi
        return center + r * np.stack([np.cos(theta), np.sin(theta)])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):  # pragma: no cover
        if self.fig is None:
            self.fig = plt.figure()
            plt.ion()
            plt.show()
        else:
            plt.figure(self.fig.number)

        plt.cla()

        # Arm
        plt.plot(self.joints[:, 0], self.joints[:, 1], 'ro-', markerfacecolor='k')

        # goal
        goal_pos = self._goal_pos.T
        plt.plot(goal_pos[0], goal_pos[1], 'gx')
        # distance between end effector and goal
        plt.plot([self.end_effector[0], goal_pos[0]], [self.end_effector[1], goal_pos[1]], 'g--')

        lim = np.sum(self.link_lengths) + 0.5
        plt.xlim([-lim, lim])
        plt.ylim([-lim, lim])
        plt.draw()
        plt.pause(0.0001)

    def close(self):
        del self.fig

    @property
    def end_effector(self):
        return self.joints[self.n_links].T


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi
