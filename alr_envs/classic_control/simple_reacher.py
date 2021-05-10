import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from gym.utils import seeding

from alr_envs.utils.utils import angle_normalize


# if os.environ.get("DISPLAY", None):
#     mpl.use('Qt5Agg')


class SimpleReacherEnv(gym.Env):
    """
    Simple Reaching Task without any physics simulation.
    Returns no reward until 150 time steps. This allows the agent to explore the space, but requires precise actions
    towards the end of the trajectory.
    """

    def __init__(self, n_links, random_start=True):
        super().__init__()
        self.link_lengths = np.ones(n_links)
        self.n_links = n_links
        self.dt = 0.1

        self.random_start = random_start

        self._goal = None

        self._joints = None
        self._joint_angle = None
        self._angle_velocity = None
        self._start_pos = None

        self.max_torque = 1  # 10
        self.steps_before_reward = 199

        action_bound = np.ones((self.n_links,))
        state_bound = np.hstack([
            [np.pi] * self.n_links,  # cos
            [np.pi] * self.n_links,  # sin
            [np.inf] * self.n_links,  # velocity
            [np.inf] * 2,  # x-y coordinates of target distance
            [np.inf]  # env steps, because reward start after n steps TODO: Maybe
        ])
        self.action_space = spaces.Box(low=-action_bound, high=action_bound, shape=action_bound.shape)
        self.observation_space = spaces.Box(low=-state_bound, high=state_bound, shape=state_bound.shape)

        self.fig = None
        self.metadata = {'render.modes': ["human"]}

        self._steps = 0
        self.seed()

    def step(self, action: np.ndarray):

        # action = self._add_action_noise(action)
        action = np.clip(action, -self.max_torque, self.max_torque)

        self._angle_velocity = self._angle_velocity + self.dt * action
        self._joint_angle = angle_normalize(self._joint_angle + self.dt * self._angle_velocity)
        self._update_joints()
        self._steps += 1

        reward, info = self._get_reward(action)

        # done = np.abs(self.end_effector - self._goal_pos) < 0.1
        done = False

        return self._get_obs().copy(), reward, done, info

    def _add_action_noise(self, action: np.ndarray):
        """
        add unobserved Gaussian Noise N(0,0.01) to the actions
        Args:
            action:

        Returns: actions with noise

        """
        return self.np_random.normal(0, 0.1, *action.shape) + action

    def _get_obs(self):
        theta = self._joint_angle
        return np.hstack([
            np.cos(theta),
            np.sin(theta),
            self._angle_velocity,
            self.end_effector - self._goal,
            self._steps
        ])

    def _update_joints(self):
        """
        update joints to get new end-effector position. The other links are only required for rendering.
        Returns:

        """
        angles = np.cumsum(self._joint_angle)
        x = self.link_lengths * np.vstack([np.cos(angles), np.sin(angles)])
        self._joints[1:] = self._joints[0] + np.cumsum(x.T, axis=0)

    def _get_reward(self, action: np.ndarray):
        diff = self.end_effector - self._goal
        reward_dist = 0

        # TODO: Is this the best option
        if self._steps >= self.steps_before_reward:
            reward_dist -= np.linalg.norm(diff)
            # reward_dist = np.exp(-0.1 * diff ** 2).mean()
            # reward_dist = - (diff ** 2).mean()

        reward_ctrl = (action ** 2).sum()
        reward = reward_dist - reward_ctrl
        return reward, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def reset(self):

        # TODO: maybe do initialisation more random?
        # Sample only orientation of first link, i.e. the arm is always straight.
        if self.random_start:
            self._joint_angle = np.hstack([[self.np_random.uniform(-np.pi, np.pi)], np.zeros(self.n_links - 1)])
        else:
            self._joint_angle = np.zeros(self.n_links)

        self._start_pos = self._joint_angle
        self._angle_velocity = np.zeros(self.n_links)
        self._joints = np.zeros((self.n_links + 1, 2))
        self._update_joints()
        self._steps = 0

        self._goal = self._get_random_goal()
        return self._get_obs().copy()

    def _get_random_goal(self):
        center = self._joints[0]

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
        plt.title(f"Iteration: {self._steps}, distance: {self.end_effector - self._goal}")

        # Arm
        plt.plot(self._joints[:, 0], self._joints[:, 1], 'ro-', markerfacecolor='k')

        # goal
        goal_pos = self._goal.T
        plt.plot(goal_pos[0], goal_pos[1], 'gx')
        # distance between end effector and goal
        plt.plot([self.end_effector[0], goal_pos[0]], [self.end_effector[1], goal_pos[1]], 'g--')

        lim = np.sum(self.link_lengths) + 0.5
        plt.xlim([-lim, lim])
        plt.ylim([-lim, lim])
        # plt.draw()
        # plt.pause(1e-4) pushes window to foreground, which is annoying.
        self.fig.canvas.flush_events()

    def close(self):
        del self.fig

    @property
    def end_effector(self):
        return self._joints[self.n_links].T
