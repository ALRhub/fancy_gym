from typing import Iterable, Union
from abc import ABCMeta, abstractmethod
import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from gym.utils import seeding
from alr_envs.classic_control.utils import check_self_collision


class BaseReacherEnv(gym.Env):
    """
    Simple Reaching Task without any physics simulation.
    Returns no reward until 150 time steps. This allows the agent to explore the space, but requires precise actions
    towards the end of the trajectory.
    """

    def __init__(self, n_links: int, random_start: bool = True,
                 allow_self_collision: bool = False, collision_penalty: float = 1000):
        super().__init__()
        self.link_lengths = np.ones(n_links)
        self.n_links = n_links
        self._dt = 0.01

        self.random_start = random_start

        self._joints = None
        self._joint_angles = None
        self._angle_velocity = None
        self._is_collided = False
        self.allow_self_collision = allow_self_collision
        self.collision_penalty = collision_penalty
        self._start_pos = np.hstack([[np.pi / 2], np.zeros(self.n_links - 1)])
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

    @property
    def current_pos(self):
        return self._joint_angles.copy()

    @property
    def current_vel(self):
        return self._angle_velocity.copy()

    def reset(self):
        # Sample only orientation of first link, i.e. the arm is always straight.
        if self.random_start:
            first_joint = self.np_random.uniform(np.pi / 4, 3 * np.pi / 4)
            self._joint_angles = np.hstack([[first_joint], np.zeros(self.n_links - 1)])
            self._start_pos = self._joint_angles.copy()
        else:
            self._joint_angles = self._start_pos

        self._angle_velocity = self._start_vel
        self._joints = np.zeros((self.n_links + 1, 2))
        self._update_joints()
        self._steps = 0

        return self._get_obs().copy()

    def step(self, action: np.ndarray):
        """
        A single step with action in torque space
        """

        # action = self._add_action_noise(action)
        ac = np.clip(action, -self.max_torque, self.max_torque)

        self._angle_velocity = self._angle_velocity + self.dt * ac
        self._joint_angles = self._joint_angles + self.dt * self._angle_velocity
        self._update_joints()

        if not self.allow_self_collision:
            self_collision = check_self_collision(line_points_in_taskspace)
            if np.any(np.abs(self._joint_angles) > np.pi) and not self.allow_self_collision:
                self_collision = True

        self._is_collided = self._check_collisions()

        reward, info = self._get_reward(action)

        self._steps += 1
        done = False

        return self._get_obs().copy(), reward, done, info

    def _update_joints(self):
        """
        update joints to get new end-effector position. The other links are only required for rendering.
        Returns:

        """
        angles = np.cumsum(self._joint_angles)
        x = self.link_lengths * np.vstack([np.cos(angles), np.sin(angles)])
        self._joints[1:] = self._joints[0] + np.cumsum(x.T, axis=0)

    @abstractmethod
    def _get_reward(self, action: np.ndarray) -> (float, dict):
        pass

    @abstractmethod
    def _get_obs(self) -> np.ndarray:
        pass

    @abstractmethod
    def _check_collisions(self) -> bool:
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        del self.fig

    @property
    def end_effector(self):
        return self._joints[self.n_links].T
