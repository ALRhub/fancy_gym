from typing import Union

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym.utils import seeding
from matplotlib import patches

from alr_envs.classic_control.utils import check_self_collision
from alr_envs.utils.mps.alr_env import AlrEnv


class HoleReacherEnv(AlrEnv):

    def __init__(self, n_links: int, hole_x: Union[None, float] = None, hole_depth: Union[None, float] = None,
                 hole_width: float = 1., random_start: bool = False, allow_self_collision: bool = False,
                 allow_wall_collision: bool = False, collision_penalty: float = 1000):

        self.n_links = n_links
        self.link_lengths = np.ones((n_links, 1))

        self.random_start = random_start

        # provided initial parameters
        self._hole_x = hole_x  # x-position of center of hole
        self._hole_width = hole_width  # width of hole
        self._hole_depth = hole_depth  # depth of hole

        # temp container for current env state
        self._tmp_hole_x = None
        self._tmp_hole_width = None
        self._tmp_hole_depth = None
        self._goal = None  # x-y coordinates for reaching the center at the bottom of the hole

        # collision
        self.allow_self_collision = allow_self_collision
        self.allow_wall_collision = allow_wall_collision
        self.collision_penalty = collision_penalty

        # state
        self._joints = None
        self._joint_angles = None
        self._angle_velocity = None
        self._start_pos = np.hstack([[np.pi / 2], np.zeros(self.n_links - 1)])
        self._start_vel = np.zeros(self.n_links)

        self.dt = 0.01

        action_bound = np.pi * np.ones((self.n_links,))
        state_bound = np.hstack([
            [np.pi] * self.n_links,  # cos
            [np.pi] * self.n_links,  # sin
            [np.inf] * self.n_links,  # velocity
            [np.inf],  # hole width
            # [np.inf],  # hole depth
            [np.inf] * 2,  # x-y coordinates of target distance
            [np.inf]  # env steps, because reward start after n steps TODO: Maybe
        ])
        self.action_space = gym.spaces.Box(low=-action_bound, high=action_bound, shape=action_bound.shape)
        self.observation_space = gym.spaces.Box(low=-state_bound, high=state_bound, shape=state_bound.shape)

        # containers for plotting
        self.metadata = {'render.modes': ["human", "partial"]}
        self.fig = None

        self._steps = 0
        self.seed()

    def step(self, action: np.ndarray):
        """
        A single step with an action in joint velocity space
        """

        acc = (action - self._angle_velocity) / self.dt
        self._angle_velocity = action
        self._joint_angles = self._joint_angles + self.dt * self._angle_velocity  # + 0.001 * np.random.randn(5)
        self._update_joints()

        reward, info = self._get_reward(acc)

        info.update({"is_collided": self._is_collided})
        self.end_effector_traj.append(np.copy(self.end_effector))

        self._steps += 1
        done = self._is_collided

        return self._get_obs().copy(), reward, done, info

    def reset(self):
        if self.random_start:
            # Maybe change more than dirst seed
            first_joint = self.np_random.uniform(np.pi / 4, 3 * np.pi / 4)
            self._joint_angles = np.hstack([[first_joint], np.zeros(self.n_links - 1)])
            self._start_pos = self._joint_angles.copy()
        else:
            self._joint_angles = self._start_pos

        self._generate_hole()
        self._set_patches()

        self._angle_velocity = self._start_vel
        self._joints = np.zeros((self.n_links + 1, 2))
        self._update_joints()
        self._steps = 0
        self.end_effector_traj = []

        return self._get_obs().copy()

    def _generate_hole(self):
        self._tmp_hole_x = self.np_random.uniform(1, 3.5, 1) if self._hole_x is None else np.copy(self._hole_x)
        self._tmp_hole_width = self.np_random.uniform(0.15, 0.5, 1) if self._hole_width is None else np.copy(
            self._hole_width)
        # TODO we do not want this right now.
        self._tmp_hole_depth = self.np_random.uniform(1, 1, 1) if self._hole_depth is None else np.copy(
            self._hole_depth)
        self._goal = np.hstack([self._tmp_hole_x, -self._tmp_hole_depth])

    def _update_joints(self):
        """
        update _joints to get new end effector position. The other links are only required for rendering.
        Returns:

        """
        line_points_in_taskspace = self._get_forward_kinematics(num_points_per_link=20)

        self._joints[1:, 0] = self._joints[0, 0] + line_points_in_taskspace[:, -1, 0]
        self._joints[1:, 1] = self._joints[0, 1] + line_points_in_taskspace[:, -1, 1]

        self_collision = False
        wall_collision = False

        if not self.allow_self_collision:
            self_collision = check_self_collision(line_points_in_taskspace)
            if np.any(np.abs(self._joint_angles) > np.pi) and not self.allow_self_collision:
                self_collision = True

        if not self.allow_wall_collision:
            wall_collision = self._check_wall_collision(line_points_in_taskspace)

        self._is_collided = self_collision or wall_collision

    def _get_reward(self, acc: np.ndarray):
        reward = 0
        # success = False

        if self._steps == 199 or self._is_collided:
            # return reward only in last time step
            # Episode also terminates when colliding, hence return reward
            dist = np.linalg.norm(self.end_effector - self._goal)
            # success = dist < 0.005 and not self._is_collided
            reward = - dist ** 2 - self.collision_penalty * self._is_collided

        reward -= 5e-8 * np.sum(acc ** 2)
        # info = {"is_success": success}

        return reward, {}  # info

    def _get_obs(self):
        theta = self._joint_angles
        return np.hstack([
            np.cos(theta),
            np.sin(theta),
            self._angle_velocity,
            self._tmp_hole_width,
            # self._tmp_hole_depth,
            self.end_effector - self._goal,
            self._steps
        ])

    def _get_forward_kinematics(self, num_points_per_link=1):
        theta = self._joint_angles[:, None]

        intermediate_points = np.linspace(0, 1, num_points_per_link) if num_points_per_link > 1 else 1
        accumulated_theta = np.cumsum(theta, axis=0)
        end_effector = np.zeros(shape=(self.n_links, num_points_per_link, 2))

        x = np.cos(accumulated_theta) * self.link_lengths * intermediate_points
        y = np.sin(accumulated_theta) * self.link_lengths * intermediate_points

        end_effector[0, :, 0] = x[0, :]
        end_effector[0, :, 1] = y[0, :]

        for i in range(1, self.n_links):
            end_effector[i, :, 0] = x[i, :] + end_effector[i - 1, -1, 0]
            end_effector[i, :, 1] = y[i, :] + end_effector[i - 1, -1, 1]

        return np.squeeze(end_effector + self._joints[0, :])

    def _check_wall_collision(self, line_points):

        # all points that are before the hole in x
        r, c = np.where(line_points[:, :, 0] < (self._tmp_hole_x - self._tmp_hole_width / 2))

        # check if any of those points are below surface
        nr_line_points_below_surface_before_hole = np.sum(line_points[r, c, 1] < 0)

        if nr_line_points_below_surface_before_hole > 0:
            return True

        # all points that are after the hole in x
        r, c = np.where(line_points[:, :, 0] > (self._tmp_hole_x + self._tmp_hole_width / 2))

        # check if any of those points are below surface
        nr_line_points_below_surface_after_hole = np.sum(line_points[r, c, 1] < 0)

        if nr_line_points_below_surface_after_hole > 0:
            return True

        # all points that are above the hole
        r, c = np.where((line_points[:, :, 0] > (self._tmp_hole_x - self._tmp_hole_width / 2)) & (
                line_points[:, :, 0] < (self._tmp_hole_x + self._tmp_hole_width / 2)))

        # check if any of those points are below surface
        nr_line_points_below_surface_in_hole = np.sum(line_points[r, c, 1] < -self._tmp_hole_depth)

        if nr_line_points_below_surface_in_hole > 0:
            return True

        return False

    def render(self, mode='human'):
        if self.fig is None:
            # Create base figure once on the beginning. Afterwards only update
            plt.ion()
            self.fig = plt.figure()
            ax = self.fig.add_subplot(1, 1, 1)

            # limits
            lim = np.sum(self.link_lengths) + 0.5
            ax.set_xlim([-lim, lim])
            ax.set_ylim([-1.1, lim])

            self.line, = ax.plot(self._joints[:, 0], self._joints[:, 1], 'ro-', markerfacecolor='k')
            self._set_patches()
            self.fig.show()

        self.fig.gca().set_title(
            f"Iteration: {self._steps}, distance: {self.end_effector - self._goal}")

        if mode == "human":

            # arm
            self.line.set_data(self._joints[:, 0], self._joints[:, 1])

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        elif mode == "partial":
            if self._steps % 20 == 0 or self._steps in [1, 199] or self._is_collided:
                # Arm
                plt.plot(self._joints[:, 0], self._joints[:, 1], 'ro-', markerfacecolor='k',
                         alpha=self._steps / 200)

    def _set_patches(self):
        if self.fig is not None:
            self.fig.gca().patches = []
            left_block = patches.Rectangle((-self.n_links, -self._tmp_hole_depth),
                                           self.n_links + self._tmp_hole_x - self._tmp_hole_width / 2,
                                           self._tmp_hole_depth,
                                           fill=True, edgecolor='k', facecolor='k')
            right_block = patches.Rectangle((self._tmp_hole_x + self._tmp_hole_width / 2, -self._tmp_hole_depth),
                                            self.n_links - self._tmp_hole_x + self._tmp_hole_width / 2,
                                            self._tmp_hole_depth,
                                            fill=True, edgecolor='k', facecolor='k')
            hole_floor = patches.Rectangle((self._tmp_hole_x - self._tmp_hole_width / 2, -self._tmp_hole_depth),
                                           self._tmp_hole_width,
                                           1 - self._tmp_hole_depth,
                                           fill=True, edgecolor='k', facecolor='k')

            # Add the patch to the Axes
            self.fig.gca().add_patch(left_block)
            self.fig.gca().add_patch(right_block)
            self.fig.gca().add_patch(hole_floor)

    @property
    def active_obs(self):
        return np.hstack([
            [self.random_start] * self.n_links,  # cos
            [self.random_start] * self.n_links,  # sin
            [self.random_start] * self.n_links,  # velocity
            [self._hole_width is None],  # hole width
            # [self._hole_depth is None],  # hole depth
            [True] * 2,  # x-y coordinates of target distance
            [False]  # env steps
        ])

    @property
    def start_pos(self) -> Union[float, int, np.ndarray]:
        return self._start_pos

    @property
    def goal_pos(self) -> Union[float, int, np.ndarray]:
        raise ValueError("Goal position is not available and has to be learnt based on the environment.")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @property
    def end_effector(self):
        return self._joints[self.n_links].T

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)


if __name__ == '__main__':
    nl = 5
    render_mode = "human"  # "human" or "partial" or "final"
    env = HoleReacherEnv(n_links=nl, allow_self_collision=False, allow_wall_collision=False, hole_width=None,
                         hole_depth=1, hole_x=None)
    obs = env.reset()

    for i in range(2000):
        # objective.load_result("/tmp/cma")
        # test with random actions
        ac = 2 * env.action_space.sample()
        obs, rew, d, info = env.step(ac)
        if i % 10 == 0:
            env.render(mode=render_mode)

        print(rew)

        if d:
            env.reset()

    env.close()
