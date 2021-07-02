from typing import Iterable, Union

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from gym.utils import seeding


class SimpleReacherEnv(gym.Env):
    """
    Simple Reaching Task without any physics simulation.
    Returns no reward until 150 time steps. This allows the agent to explore the space, but requires precise actions
    towards the end of the trajectory.
    """

    def __init__(self, n_links: int, target: Union[None, Iterable] = None, random_start: bool = True):
        super().__init__()
        self.link_lengths = np.ones(n_links)
        self.n_links = n_links
        self._dt = 0.1

        self.random_start = random_start

        # provided initial parameters
        self.inital_target = target

        # temp container for current env state
        self._goal = None

        self._joints = None
        self._joint_angles = None
        self._angle_velocity = None
        self._start_pos = np.zeros(self.n_links)
        self._start_vel = np.zeros(self.n_links)

        self.max_torque = 1
        self.steps_before_reward = 199

        action_bound = np.ones((self.n_links,)) * self.max_torque
        state_bound = np.hstack([
            [np.pi] * self.n_links,  # cos
            [np.pi] * self.n_links,  # sin
            [np.inf] * self.n_links,  # velocity
            [np.inf] * 2,  # x-y coordinates of target distance
            [np.inf]  # env steps, because reward start after n steps TODO: Maybe
        ])
        self.action_space = spaces.Box(low=-action_bound, high=action_bound, shape=action_bound.shape)
        self.observation_space = spaces.Box(low=-state_bound, high=state_bound, shape=state_bound.shape)

        # containers for plotting
        self.metadata = {'render.modes': ["human"]}
        self.fig = None

        self._steps = 0
        self.seed()

    @property
    def dt(self) -> Union[float, int]:
        return self._dt

    # @property
    # def start_pos(self):
    #     return self._start_pos

    @property
    def current_pos(self):
        return self._joint_angles

    @property
    def current_vel(self):
        return self._angle_velocity

    def step(self, action: np.ndarray):
        """
        A single step with action in torque space
        """

        # action = self._add_action_noise(action)
        ac = np.clip(action, -self.max_torque, self.max_torque)

        self._angle_velocity = self._angle_velocity + self.dt * ac
        self._joint_angles = self._joint_angles + self.dt * self._angle_velocity
        self._update_joints()

        reward, info = self._get_reward(action)

        self._steps += 1
        done = False

        return self._get_obs().copy(), reward, done, info

    def reset(self):

        # TODO: maybe do initialisation more random?
        # Sample only orientation of first link, i.e. the arm is always straight.
        if self.random_start:
            self._joint_angles = np.hstack([[self.np_random.uniform(-np.pi, np.pi)], np.zeros(self.n_links - 1)])
            self._start_pos = self._joint_angles.copy()
        else:
            self._joint_angles = self._start_pos

        self._generate_goal()

        self._angle_velocity = self._start_vel
        self._joints = np.zeros((self.n_links + 1, 2))
        self._update_joints()
        self._steps = 0

        return self._get_obs().copy()

    def _update_joints(self):
        """
        update joints to get new end-effector position. The other links are only required for rendering.
        Returns:

        """
        angles = np.cumsum(self._joint_angles)
        x = self.link_lengths * np.vstack([np.cos(angles), np.sin(angles)])
        self._joints[1:] = self._joints[0] + np.cumsum(x.T, axis=0)

    def _get_reward(self, action: np.ndarray):
        diff = self.end_effector - self._goal
        reward_dist = 0

        if self._steps >= self.steps_before_reward:
            reward_dist -= np.linalg.norm(diff)
            # reward_dist = np.exp(-0.1 * diff ** 2).mean()
            # reward_dist = - (diff ** 2).mean()

        reward_ctrl = (action ** 2).sum()
        reward = reward_dist - reward_ctrl
        return reward, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        theta = self._joint_angles
        return np.hstack([
            np.cos(theta),
            np.sin(theta),
            self._angle_velocity,
            self.end_effector - self._goal,
            self._steps
        ])

    def _generate_goal(self):

        if self.inital_target is None:

            total_length = np.sum(self.link_lengths)
            goal = np.array([total_length, total_length])
            while np.linalg.norm(goal) >= total_length:
                goal = self.np_random.uniform(low=-total_length, high=total_length, size=2)
        else:
            goal = np.copy(self.inital_target)

        self._goal = goal

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

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        del self.fig

    @property
    def end_effector(self):
        return self._joints[self.n_links].T
