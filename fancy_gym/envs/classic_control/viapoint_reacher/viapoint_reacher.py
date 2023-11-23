from typing import Iterable, Union, Tuple, Optional, Any, Dict

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from gymnasium.core import ObsType

from fancy_gym.envs.classic_control.base_reacher.base_reacher_direct import BaseReacherDirectEnv
from . import MPWrapper


class ViaPointReacherEnv(BaseReacherDirectEnv):

    def __init__(self, n_links, random_start: bool = False, via_target: Union[None, Iterable] = None,
                 target: Union[None, Iterable] = None, allow_self_collision=False, collision_penalty=1000, **kwargs):

        super().__init__(n_links, random_start, allow_self_collision, **kwargs)

        # provided initial parameters
        self.intitial_target = target  # provided target value
        self.initial_via_target = via_target  # provided via point target value

        # temp container for current env state
        self._via_point = np.ones(2)
        self._goal = np.array((n_links, 0))

        # collision
        self.collision_penalty = collision_penalty

        state_bound = np.hstack([
            [np.pi] * self.n_links,  # cos
            [np.pi] * self.n_links,  # sin
            [np.inf] * self.n_links,  # velocity
            [np.inf] * 2,  # x-y coordinates of via point distance
            [np.inf] * 2,  # x-y coordinates of target distance
            [np.inf]  # env steps, because reward start after n steps
        ])
        self.observation_space = spaces.Box(low=-state_bound, high=state_bound, shape=state_bound.shape)

    # @property
    # def start_pos(self):
    #     return self._start_pos

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) \
            -> Tuple[ObsType, Dict[str, Any]]:
        # Reset twice to ensure we return obs after generating goal and generating goal after executing seeded reset.
        # (Env will not behave deterministic otherwise)
        # Yes, there is probably a more elegant solution to this problem...
        self._generate_goal()
        super().reset(seed=seed, options=options)
        self._generate_goal()
        return super().reset(seed=seed, options=options)

    def _generate_goal(self):
        # TODO: Maybe improve this later, this can yield quite a lot of invalid settings

        total_length = np.sum(self.link_lengths)

        # rejection sampled point in inner circle with 0.5*Radius
        if self.initial_via_target is None:
            via_target = np.array([total_length, total_length])
            while np.linalg.norm(via_target) >= 0.5 * total_length:
                via_target = self.np_random.uniform(low=-0.5 * total_length, high=0.5 * total_length, size=2)
        else:
            via_target = np.copy(self.initial_via_target)

        # rejection sampled point in outer circle
        if self.intitial_target is None:
            goal = np.array([total_length, total_length])
            while np.linalg.norm(goal) >= total_length or np.linalg.norm(goal) <= 0.5 * total_length:
                goal = self.np_random.uniform(low=-total_length, high=total_length, size=2)
        else:
            goal = np.copy(self.intitial_target)

        self._via_point = via_target
        self._goal = goal

    def _get_reward(self, acc):
        success = False
        reward = -np.inf

        if not self.allow_self_collision:
            self._is_collided = self._check_self_collision()

        if not self._is_collided:
            dist = np.inf
            # return intermediate reward for via point
            if self._steps == 100:
                dist = np.linalg.norm(self.end_effector - self._via_point)
            # return reward in last time step for goal
            elif self._steps == 199:
                dist = np.linalg.norm(self.end_effector - self._goal)

            success = dist < 0.005
        else:
            # Episode terminates when colliding, hence return reward
            dist = np.linalg.norm(self.end_effector - self._goal)
            reward = -self.collision_penalty

        reward -= dist ** 2
        reward -= 5e-8 * np.sum(acc ** 2)
        info = {"is_success": success,
                "is_collided": self._is_collided,
                "end_effector": np.copy(self.end_effector)}

        return reward, info

    def _terminate(self, info):
        return info["is_collided"]

    def _get_obs(self):
        theta = self._joint_angles
        return np.hstack([
            np.cos(theta),
            np.sin(theta),
            self._angle_velocity,
            self.end_effector - self._via_point,
            self.end_effector - self._goal,
            self._steps
        ]).astype(np.float32)

    def _check_collisions(self) -> bool:
        return self._check_self_collision()

    def render(self):
        goal_pos = self._goal.T
        via_pos = self._via_point.T

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
            self.goal_point_plot, = ax.plot(goal_pos[0], goal_pos[1], 'go')
            self.via_point_plot, = ax.plot(via_pos[0], via_pos[1], 'gx')

            self.fig.show()

        self.fig.gca().set_title(f"Iteration: {self._steps}, distance: {self.end_effector - self._goal}")

        if self.render_mode == "human":
            # goal
            if self._steps == 1:
                self.goal_point_plot.set_data(goal_pos[0], goal_pos[1])
                self.via_point_plot.set_data(via_pos[0], goal_pos[1])

            # arm
            self.line.set_data(self._joints[:, 0], self._joints[:, 1])

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        elif self.render_mode == "partial":
            if self._steps == 1:
                # fig, ax = plt.subplots()
                # Add the patch to the Axes
                [plt.gca().add_patch(rect) for rect in self.patches]
                # plt.pause(0.01)

            if self._steps % 20 == 0 or self._steps in [1, 199] or self._is_collided:
                # Arm
                plt.plot(self._joints[:, 0], self._joints[:, 1], 'ro-', markerfacecolor='k', alpha=self._steps / 200)
                # ax.plot(line_points_in_taskspace[:, 0, 0],
                #         line_points_in_taskspace[:, 0, 1],
                #         line_points_in_taskspace[:, -1, 0],
                #         line_points_in_taskspace[:, -1, 1], marker='o', color='k', alpha=t / 200)

                lim = np.sum(self.link_lengths) + 0.5
                plt.xlim([-lim, lim])
                plt.ylim([-1.1, lim])
                plt.pause(0.01)

        elif self.render_mode == "final":
            if self._steps == 199 or self._is_collided:
                # fig, ax = plt.subplots()

                # Add the patch to the Axes
                [plt.gca().add_patch(rect) for rect in self.patches]

                plt.xlim(-self.n_links, self.n_links), plt.ylim(-1, self.n_links)
                # Arm
                plt.plot(self._joints[:, 0], self._joints[:, 1], 'ro-', markerfacecolor='k')

                plt.pause(0.01)
