import gym
import matplotlib.pyplot as plt
import numpy as np

from alr_envs import DmpWrapper
from alr_envs.utils.utils import check_self_collision


def viapoint_dmp(**kwargs):
    _env = gym.make("alr_envs:ViaPointReacher-v0")
    # _env = ViaPointReacher(**kwargs)
    return DmpWrapper(_env, num_dof=5, num_basis=5, duration=2, alpha_phase=2.5, dt=_env.dt,
                      start_pos=_env.start_pos, learn_goal=False, policy_type="velocity", weights_scale=50)


class ViaPointReacher(gym.Env):

    def __init__(self, n_links, allow_self_collision=False, collision_penalty=1000):
        self.num_links = n_links
        self.link_lengths = np.ones((n_links, 1))

        # task
        self.via_point = np.ones(2)
        self.goal_point = np.array((n_links, 0))

        # collision
        self.allow_self_collision = allow_self_collision
        self.collision_penalty = collision_penalty

        # state
        self._joints = None
        self._joint_angles = None
        self._angle_velocity = None
        self.start_pos = np.hstack([[np.pi / 2], np.zeros(self.num_links - 1)])
        self.start_vel = np.zeros(self.num_links)
        self.weight_matrix_scale = 1

        self._steps = 0
        self.dt = 0.01
        # self.time_limit = 2

        action_bound = np.pi * np.ones((self.num_links,))
        state_bound = np.hstack([
            [np.pi] * self.num_links,  # cos
            [np.pi] * self.num_links,  # sin
            [np.inf] * self.num_links,  # velocity
            [np.inf] * 2,  # x-y coordinates of target distance
            [np.inf]  # env steps, because reward start after n steps TODO: Maybe
        ])
        self.action_space = gym.spaces.Box(low=-action_bound, high=action_bound, shape=action_bound.shape)
        self.observation_space = gym.spaces.Box(low=-state_bound, high=state_bound, shape=state_bound.shape)

        self.fig = None

    @property
    def end_effector(self):
        return self._joints[self.num_links].T

    def configure(self, context):
        pass

    def reset(self):
        self._joint_angles = self.start_pos
        self._angle_velocity = self.start_vel
        self._joints = np.zeros((self.num_links + 1, 2))
        self._update_joints()
        self._steps = 0

        return self._get_obs().copy()

    def step(self, action: np.ndarray):
        """
        a single step with an action in joint velocity space
        """
        vel = action
        acc = (vel - self._angle_velocity) / self.dt
        self._angle_velocity = vel
        self._joint_angles = self._joint_angles + self.dt * self._angle_velocity

        self._update_joints()

        dist_reward = 0
        if not self._is_collided:
            if self._steps == 100:
                dist_reward = np.linalg.norm(self.end_effector - self.via_point)
            elif self._steps == 199:
                dist_reward = np.linalg.norm(self.end_effector - self.goal_point)

        # TODO: Do we need that?
        reward = - dist_reward ** 2

        reward -= 5e-8 * np.sum(acc**2)

        if self._is_collided:
            reward -= self.collision_penalty

        info = {"is_collided": self._is_collided}

        self._steps += 1

        # done = self._steps * self.dt > self.time_limit or self._is_collided
        done = self._is_collided

        return self._get_obs().copy(), reward, done, info

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

    def _get_obs(self):
        theta = self._joint_angles
        return np.hstack([
            np.cos(theta),
            np.sin(theta),
            self._angle_velocity,
            self.end_effector - self.via_point,
            self.end_effector - self.goal_point,
            self._steps
        ])

    def get_forward_kinematics(self, num_points_per_link=1):
        theta = self._joint_angles[:, None]

        intermediate_points = np.linspace(0, 1, num_points_per_link) if num_points_per_link > 1 else 1

        accumulated_theta = np.cumsum(theta, axis=0)

        endeffector = np.zeros(shape=(self.num_links, num_points_per_link, 2))

        x = np.cos(accumulated_theta) * self.link_lengths * intermediate_points
        y = np.sin(accumulated_theta) * self.link_lengths * intermediate_points

        endeffector[0, :, 0] = x[0, :]
        endeffector[0, :, 1] = y[0, :]

        for i in range(1, self.num_links):
            endeffector[i, :, 0] = x[i, :] + endeffector[i - 1, -1, 0]
            endeffector[i, :, 1] = y[i, :] + endeffector[i - 1, -1, 1]

        return np.squeeze(endeffector + self._joints[0, :])

    def render(self, mode='human'):
        if self.fig is None:
            self.fig = plt.figure()
            # plt.ion()
            # plt.pause(0.01)
        else:
            plt.figure(self.fig.number)

        if mode == "human":
            plt.cla()
            plt.title(f"Iteration: {self._steps}")

            # Arm
            plt.plot(self._joints[:, 0], self._joints[:, 1], 'ro-', markerfacecolor='k')

            lim = np.sum(self.link_lengths) + 0.5
            plt.xlim([-lim, lim])
            plt.ylim([-lim, lim])
            # plt.draw()
            plt.pause(1e-4)  # pushes window to foreground, which is annoying.
            # self.fig.canvas.flush_events()

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

                plt.xlim(-self.num_links, self.num_links), plt.ylim(-1, self.num_links)
                # Arm
                plt.plot(self._joints[:, 0], self._joints[:, 1], 'ro-', markerfacecolor='k')

                plt.pause(0.01)

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
