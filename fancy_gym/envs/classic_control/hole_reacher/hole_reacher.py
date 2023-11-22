from typing import Union, Optional, Tuple, Any, Dict

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from gymnasium.core import ObsType
from matplotlib import patches

from fancy_gym.envs.classic_control.base_reacher.base_reacher_direct import BaseReacherDirectEnv
from . import MPWrapper

MAX_EPISODE_STEPS_HOLEREACHER = 200


class HoleReacherEnv(BaseReacherDirectEnv):

    def __init__(self, n_links: int, hole_x: Union[None, float] = None, hole_depth: Union[None, float] = None,
                 hole_width: float = 1., random_start: bool = False, allow_self_collision: bool = False,
                 allow_wall_collision: bool = False, collision_penalty: float = 1000, rew_fct: str = "simple", **kwargs):

        super().__init__(n_links, random_start, allow_self_collision, **kwargs)

        # provided initial parameters
        self.initial_x = hole_x  # x-position of center of hole
        self.initial_width = hole_width  # width of hole
        self.initial_depth = hole_depth  # depth of hole

        # temp container for current env state
        self._tmp_x = None
        self._tmp_width = None
        self._tmp_depth = None
        self._goal = None  # x-y coordinates for reaching the center at the bottom of the hole

        # action_bound = np.pi * np.ones((self.n_links,))
        state_bound = np.hstack([
            [np.pi] * self.n_links,  # cos
            [np.pi] * self.n_links,  # sin
            [np.inf] * self.n_links,  # velocity
            [np.inf],  # hole width
            # [np.inf],  # hole depth
            [np.inf] * 2,  # x-y coordinates of target distance
            [np.inf]  # env steps, because reward start after n steps TODO: Maybe
        ])
        # self.action_space = gym.spaces.Box(low=-action_bound, high=action_bound, shape=action_bound.shape)
        self.observation_space = spaces.Box(low=-state_bound, high=state_bound, shape=state_bound.shape)

        if rew_fct == "simple":
            from fancy_gym.envs.classic_control.hole_reacher.hr_simple_reward import HolereacherReward
            self.reward_function = HolereacherReward(allow_self_collision, allow_wall_collision, collision_penalty)
        elif rew_fct == "vel_acc":
            from fancy_gym.envs.classic_control.hole_reacher.hr_dist_vel_acc_reward import HolereacherReward
            self.reward_function = HolereacherReward(allow_self_collision, allow_wall_collision, collision_penalty)
        elif rew_fct == "unbounded":
            from fancy_gym.envs.classic_control.hole_reacher.hr_unbounded_reward import HolereacherReward
            self.reward_function = HolereacherReward(allow_self_collision, allow_wall_collision)
        else:
            raise ValueError("Unknown reward function {}".format(rew_fct))

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) \
            -> Tuple[ObsType, Dict[str, Any]]:

        # initialize seed here as the random goal needs to be generated before the super reset()
        gym.Env.reset(self, seed=seed, options=options)

        self._generate_hole()
        self._set_patches()
        self.reward_function.reset()

        # do not provide seed to avoid setting it twice
        return super(HoleReacherEnv, self).reset(options=options)

    def _get_reward(self, action: np.ndarray) -> (float, dict):
        return self.reward_function.get_reward(self)

    def _terminate(self, info):
        return info["is_collided"]

    def _generate_hole(self):
        if self.initial_width is None:
            width = self.np_random.uniform(0.15, 0.5)
        else:
            width = np.copy(self.initial_width)
        if self.initial_x is None:
            # sample whole on left or right side
            direction = self.np_random.choice([-1, 1])
            # Hole center needs to be half the width away from the arm to give a valid setting.
            x = direction * self.np_random.uniform(width / 2, 3.5)
        else:
            x = np.copy(self.initial_x)
        if self.initial_depth is None:
            # TODO we do not want this right now.
            depth = self.np_random.uniform(1, 1)
        else:
            depth = np.copy(self.initial_depth)

        self._tmp_width = width
        self._tmp_x = x
        self._tmp_depth = depth
        self._goal = np.hstack([self._tmp_x, -self._tmp_depth])

        self._line_ground_left = np.array([-self.n_links, 0, x - width / 2, 0])
        self._line_ground_right = np.array([x + width / 2, 0, self.n_links, 0])
        self._line_ground_hole = np.array([x - width / 2, -depth, x + width / 2, -depth])
        self._line_hole_left = np.array([x - width / 2, -depth, x - width / 2, 0])
        self._line_hole_right = np.array([x + width / 2, -depth, x + width / 2, 0])

        self.ground_lines = np.stack((self._line_ground_left,
                                      self._line_ground_right,
                                      self._line_ground_hole,
                                      self._line_hole_left,
                                      self._line_hole_right))

    def _get_obs(self):
        theta = self._joint_angles
        return np.hstack([
            np.cos(theta),
            np.sin(theta),
            self._angle_velocity,
            self._tmp_width,
            # self._tmp_hole_depth,
            self.end_effector - self._goal,
            self._steps
        ]).astype(np.float32)

    def _get_line_points(self, num_points_per_link=1):
        theta = self._joint_angles[:, None]

        intermediate_points = np.linspace(0, 1, num_points_per_link) if num_points_per_link > 1 else 1
        accumulated_theta = np.cumsum(theta, axis=0)
        end_effector = np.zeros(shape=(self.n_links, num_points_per_link, 2))

        x = np.cos(accumulated_theta) * self.link_lengths[:, None] * intermediate_points
        y = np.sin(accumulated_theta) * self.link_lengths[:, None] * intermediate_points

        end_effector[0, :, 0] = x[0, :]
        end_effector[0, :, 1] = y[0, :]

        for i in range(1, self.n_links):
            end_effector[i, :, 0] = x[i, :] + end_effector[i - 1, -1, 0]
            end_effector[i, :, 1] = y[i, :] + end_effector[i - 1, -1, 1]

        return np.squeeze(end_effector + self._joints[0, :])

    def _check_collisions(self) -> bool:
        return self._check_self_collision() or self.check_wall_collision()

    def check_wall_collision(self):
        line_points = self._get_line_points(num_points_per_link=100)

        # all points that are before the hole in x
        r, c = np.where(line_points[:, :, 0] < (self._tmp_x - self._tmp_width / 2))

        # check if any of those points are below surface
        nr_line_points_below_surface_before_hole = np.sum(line_points[r, c, 1] < 0)

        if nr_line_points_below_surface_before_hole > 0:
            return True

        # all points that are after the hole in x
        r, c = np.where(line_points[:, :, 0] > (self._tmp_x + self._tmp_width / 2))

        # check if any of those points are below surface
        nr_line_points_below_surface_after_hole = np.sum(line_points[r, c, 1] < 0)

        if nr_line_points_below_surface_after_hole > 0:
            return True

        # all points that are above the hole
        r, c = np.where((line_points[:, :, 0] > (self._tmp_x - self._tmp_width / 2)) & (
            line_points[:, :, 0] < (self._tmp_x + self._tmp_width / 2)))

        # check if any of those points are below surface
        nr_line_points_below_surface_in_hole = np.sum(line_points[r, c, 1] < -self._tmp_depth)

        if nr_line_points_below_surface_in_hole > 0:
            return True

        return False

    def render(self):
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
            f"Iteration: {self._steps}, distance: {np.linalg.norm(self.end_effector - self._goal) ** 2}")

        if self.render_mode == "human":

            # arm
            self.line.set_data(self._joints[:, 0], self._joints[:, 1])

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        elif self.render_mode == "partial":
            if self._steps % 20 == 0 or self._steps in [1, 199] or self._is_collided:
                # Arm
                plt.plot(self._joints[:, 0], self._joints[:, 1], 'ro-', markerfacecolor='k',
                         alpha=self._steps / 200)

    def _set_patches(self):
        if self.fig is not None:
            # self.fig.gca().patches = []
            left_block = patches.Rectangle((-self.n_links, -self._tmp_depth),
                                           self.n_links + self._tmp_x - self._tmp_width / 2,
                                           self._tmp_depth,
                                           fill=True, edgecolor='k', facecolor='k')
            right_block = patches.Rectangle((self._tmp_x + self._tmp_width / 2, -self._tmp_depth),
                                            self.n_links - self._tmp_x + self._tmp_width / 2,
                                            self._tmp_depth,
                                            fill=True, edgecolor='k', facecolor='k')
            hole_floor = patches.Rectangle((self._tmp_x - self._tmp_width / 2, -self._tmp_depth),
                                           self._tmp_width,
                                           1 - self._tmp_depth,
                                           fill=True, edgecolor='k', facecolor='k')

            # Add the patch to the Axes
            self.fig.gca().add_patch(left_block)
            self.fig.gca().add_patch(right_block)
            self.fig.gca().add_patch(hole_floor)
