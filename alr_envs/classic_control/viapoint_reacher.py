from typing import Iterable, Union

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym.utils import seeding

from alr_envs.classic_control.utils import check_self_collision
from alr_envs.utils.mps.mp_environments import MPEnv


class ViaPointReacher(MPEnv):

    def __init__(self, n_links, random_start: bool = True, via_target: Union[None, Iterable] = None,
                 target: Union[None, Iterable] = None, allow_self_collision=False, collision_penalty=1000):

        self.n_links = n_links
        self.link_lengths = np.ones((n_links, 1))

        self.random_start = random_start

        # provided initial parameters
        self._target = target  # provided target value
        self._via_target = via_target  # provided via point target value

        # temp container for current env state
        self._via_point = np.ones(2)
        self._goal = np.array((n_links, 0))

        # collision
        self.allow_self_collision = allow_self_collision
        self.collision_penalty = collision_penalty

        # state
        self._joints = None
        self._joint_angles = None
        self._angle_velocity = None
        self._start_pos = np.hstack([[np.pi / 2], np.zeros(self.n_links - 1)])
        self._start_vel = np.zeros(self.n_links)
        self.weight_matrix_scale = 1

        self.dt = 0.01

        action_bound = np.pi * np.ones((self.n_links,))
        state_bound = np.hstack([
            [np.pi] * self.n_links,  # cos
            [np.pi] * self.n_links,  # sin
            [np.inf] * self.n_links,  # velocity
            [np.inf] * 2,  # x-y coordinates of via point distance
            [np.inf] * 2,  # x-y coordinates of target distance
            [np.inf]  # env steps, because reward start after n steps
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
        a single step with an action in joint velocity space
        """
        vel = action
        self._angle_velocity = vel
        self._joint_angles = self._joint_angles + self.dt * self._angle_velocity
        self._update_joints()

        acc = (vel - self._angle_velocity) / self.dt
        reward, info = self._get_reward(acc)

        info.update({"is_collided": self._is_collided})

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

        self._generate_goal()

        self._angle_velocity = self._start_vel
        self._joints = np.zeros((self.n_links + 1, 2))
        self._update_joints()
        self._steps = 0

        return self._get_obs().copy()

    def _generate_goal(self):
        # TODO: Maybe improve this later, this can yield quite a lot of invalid settings

        total_length = np.sum(self.link_lengths)

        # rejection sampled point in inner circle with 0.5*Radius
        if self._via_target is None:
            via_target = np.array([total_length, total_length])
            while np.linalg.norm(via_target) >= 0.5 * total_length:
                via_target = self.np_random.uniform(low=-0.5 * total_length, high=0.5 * total_length, size=2)
        else:
            via_target = np.copy(self._via_target)

        # rejection sampled point in outer circle
        if self._target is None:
            goal = np.array([total_length, total_length])
            while np.linalg.norm(goal) >= total_length or np.linalg.norm(goal) <= 0.5 * total_length:
                goal = self.np_random.uniform(low=-total_length, high=total_length, size=2)
        else:
            goal = np.copy(self._target)

        self._via_target = via_target
        self._goal = goal

    def _update_joints(self):
        """
        update _joints to get new end effector position. The other links are only required for rendering.
        Returns:

        """
        line_points_in_taskspace = self.get_forward_kinematics(num_points_per_link=20)

        self._joints[1:, 0] = self._joints[0, 0] + line_points_in_taskspace[:, -1, 0]
        self._joints[1:, 1] = self._joints[0, 1] + line_points_in_taskspace[:, -1, 1]

        self_collision = False

        if not self.allow_self_collision:
            self_collision = check_self_collision(line_points_in_taskspace)
            if np.any(np.abs(self._joint_angles) > np.pi):
                self_collision = True

        self._is_collided = self_collision

    def _get_reward(self, acc):
        success = False
        reward = -np.inf
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
        info = {"is_success": success}

        return reward, info

    def _get_obs(self):
        theta = self._joint_angles
        return np.hstack([
            np.cos(theta),
            np.sin(theta),
            self._angle_velocity,
            self.end_effector - self._via_point,
            self.end_effector - self._goal,
            self._steps
        ])

    def get_forward_kinematics(self, num_points_per_link=1):
        theta = self._joint_angles[:, None]

        intermediate_points = np.linspace(0, 1, num_points_per_link) if num_points_per_link > 1 else 1

        accumulated_theta = np.cumsum(theta, axis=0)

        endeffector = np.zeros(shape=(self.n_links, num_points_per_link, 2))

        x = np.cos(accumulated_theta) * self.link_lengths * intermediate_points
        y = np.sin(accumulated_theta) * self.link_lengths * intermediate_points

        endeffector[0, :, 0] = x[0, :]
        endeffector[0, :, 1] = y[0, :]

        for i in range(1, self.n_links):
            endeffector[i, :, 0] = x[i, :] + endeffector[i - 1, -1, 0]
            endeffector[i, :, 1] = y[i, :] + endeffector[i - 1, -1, 1]

        return np.squeeze(endeffector + self._joints[0, :])

    def render(self, mode='human'):
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

        if mode == "human":
            # goal
            if self._steps == 1:
                self.goal_point_plot.set_data(goal_pos[0], goal_pos[1])
                self.via_point_plot.set_data(via_pos[0], goal_pos[1])

            # arm
            self.line.set_data(self._joints[:, 0], self._joints[:, 1])

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        elif mode == "partial":
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

        elif mode == "final":
            if self._steps == 199 or self._is_collided:
                # fig, ax = plt.subplots()

                # Add the patch to the Axes
                [plt.gca().add_patch(rect) for rect in self.patches]

                plt.xlim(-self.n_links, self.n_links), plt.ylim(-1, self.n_links)
                # Arm
                plt.plot(self._joints[:, 0], self._joints[:, 1], 'ro-', markerfacecolor='k')

                plt.pause(0.01)

    @property
    def active_obs(self):
        return np.hstack([
            [self.random_start] * self.n_links,  # cos
            [self.random_start] * self.n_links,  # sin
            [self.random_start] * self.n_links,  # velocity
            [self._via_target is None] * 2,  # x-y coordinates of via point distance
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
    env = ViaPointReacher(n_links=nl, allow_self_collision=False)
    env.reset()
    env.render(mode=render_mode)

    for i in range(300):
        # objective.load_result("/tmp/cma")
        # test with random actions
        ac = env.action_space.sample()
        # ac[0] += np.pi/2
        obs, rew, d, info = env.step(ac)
        env.render(mode=render_mode)

        print(rew)

        if d:
            break

    env.close()
