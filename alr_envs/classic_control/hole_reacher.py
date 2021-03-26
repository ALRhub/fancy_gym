import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

from alr_envs import DmpWrapper
from alr_envs.utils.wrapper.detpmp_wrapper import DetPMPWrapper


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) - (B[1] - A[1]) * (C[0] - A[0]) > 1e-12


# Return true if line segments AB and CD intersect
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def holereacher_dmp(**kwargs):
    _env = gym.make("alr_envs:HoleReacher-v0")
    # _env = HoleReacher(**kwargs)
    return DmpWrapper(_env, num_dof=5, num_basis=5, duration=2, dt=_env.dt, learn_goal=True, alpha_phase=3.5,
                      start_pos=_env.start_pos, policy_type="velocity", weights_scale=100, goal_scale=0.1)


def holereacher_fix_goal_dmp(**kwargs):
    _env = gym.make("alr_envs:HoleReacher-v0")
    # _env = HoleReacher(**kwargs)
    return DmpWrapper(_env, num_dof=5, num_basis=5, duration=2, dt=_env.dt, learn_goal=False, alpha_phase=3.5,
                      start_pos=_env.start_pos, policy_type="velocity", weights_scale=50, goal_scale=1,
                      final_pos=np.array([2.02669572, -1.25966385, -1.51618198, -0.80946476, 0.02012344]))


def holereacher_detpmp(**kwargs):
    _env = gym.make("alr_envs:HoleReacher-v0")
    # _env = HoleReacher(**kwargs)
    return DetPMPWrapper(_env, num_dof=5, num_basis=5, width=0.005, policy_type="velocity", start_pos=_env.start_pos,
                         duration=2, post_traj_time=0, dt=_env.dt, weights_scale=0.25, zero_start=True, zero_goal=False)


class HoleReacher(gym.Env):

    def __init__(self, n_links, hole_x, hole_width, hole_depth, allow_self_collision=False,
                 allow_wall_collision=False, collision_penalty=1000):

        self.n_links = n_links
        self.link_lengths = np.ones((n_links, 1))

        # task
        self.hole_x = hole_x  # x-position of center of hole
        self.hole_width = hole_width  # width of hole
        self.hole_depth = hole_depth  # depth of hole

        self.bottom_center_of_hole = np.hstack([hole_x, -hole_depth])
        self.top_center_of_hole = np.hstack([hole_x, 0])
        self.left_wall_edge = np.hstack([hole_x - self.hole_width / 2, 0])
        self.right_wall_edge = np.hstack([hole_x + self.hole_width / 2, 0])

        # collision
        self.allow_self_collision = allow_self_collision
        self.allow_wall_collision = allow_wall_collision
        self.collision_penalty = collision_penalty

        # state
        self._joints = None
        self._joint_angles = None
        self._angle_velocity = None
        self.start_pos = np.hstack([[np.pi / 2], np.zeros(self.n_links - 1)])
        self.start_vel = np.zeros(self.n_links)

        self.dt = 0.01
        # self.time_limit = 2

        action_bound = np.pi * np.ones((self.n_links,))
        state_bound = np.hstack([
            [np.pi] * self.n_links,  # cos
            [np.pi] * self.n_links,  # sin
            [np.inf] * self.n_links,  # velocity
            [np.inf] * 2,  # x-y coordinates of target distance
            [np.inf]  # env steps, because reward start after n steps TODO: Maybe
        ])
        self.action_space = gym.spaces.Box(low=-action_bound, high=action_bound, shape=action_bound.shape)
        self.observation_space = gym.spaces.Box(low=-state_bound, high=state_bound, shape=state_bound.shape)

        self.fig = None
        rect_1 = patches.Rectangle((-self.n_links, -1),
                                   self.n_links + self.hole_x - self.hole_width / 2, 1,
                                   fill=True, edgecolor='k', facecolor='k')
        rect_2 = patches.Rectangle((self.hole_x + self.hole_width / 2, -1),
                                   self.n_links - self.hole_x + self.hole_width / 2, 1,
                                   fill=True, edgecolor='k', facecolor='k')
        rect_3 = patches.Rectangle((self.hole_x - self.hole_width / 2, -1), self.hole_width,
                                   1 - self.hole_depth,
                                   fill=True, edgecolor='k', facecolor='k')

        self.patches = [rect_1, rect_2, rect_3]

    @property
    def end_effector(self):
        return self._joints[self.n_links].T

    def configure(self, context):
        pass

    def reset(self):
        self._joint_angles = self.start_pos
        self._angle_velocity = self.start_vel
        self._joints = np.zeros((self.n_links + 1, 2))
        self._update_joints()
        self._steps = 0

        return self._get_obs().copy()

    def step(self, action: np.ndarray):
        """
        a single step with an action in joint velocity space
        """
        vel = action  # + 0.01 * np.random.randn(self.num_links)
        acc = (vel - self._angle_velocity) / self.dt
        self._angle_velocity = vel
        self._joint_angles = self._joint_angles + self.dt * self._angle_velocity

        self._update_joints()

        reward = 0
        if self._is_collided:
            reward = -self.collision_penalty
        else:
            if self._steps == 199:
                reward = - np.linalg.norm(self.end_effector - self.bottom_center_of_hole) ** 2

        reward -= 5e-8 * np.sum(acc ** 2)

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
        wall_collision = False

        if not self.allow_self_collision:
            self_collision = self.check_self_collision(line_points_in_taskspace)
            if np.any(np.abs(self._joint_angles) > np.pi) and not self.allow_self_collision:
                self_collision = True

        if not self.allow_wall_collision:
            wall_collision = self.check_wall_collision(line_points_in_taskspace)

        self._is_collided = self_collision or wall_collision

    def _get_obs(self):
        theta = self._joint_angles
        return np.hstack([
            np.cos(theta),
            np.sin(theta),
            self._angle_velocity,
            self.end_effector - self.bottom_center_of_hole,
            self._steps
        ])

    def get_forward_kinematics(self, num_points_per_link=1):
        theta = self._joint_angles[:, None]

        if num_points_per_link > 1:
            intermediate_points = np.linspace(0, 1, num_points_per_link)
        else:
            intermediate_points = 1

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

    def check_self_collision(self, line_points):
        for i, line1 in enumerate(line_points):
            for line2 in line_points[i + 2:, :, :]:
                # if line1 != line2:
                if intersect(line1[0], line1[-1], line2[0], line2[-1]):
                    return True
        return False

    def check_wall_collision(self, line_points):

        # all points that are before the hole in x
        r, c = np.where(line_points[:, :, 0] < (self.hole_x - self.hole_width / 2))

        # check if any of those points are below surface
        nr_line_points_below_surface_before_hole = np.sum(line_points[r, c, 1] < 0)

        if nr_line_points_below_surface_before_hole > 0:
            return True

        # all points that are after the hole in x
        r, c = np.where(line_points[:, :, 0] > (self.hole_x + self.hole_width / 2))

        # check if any of those points are below surface
        nr_line_points_below_surface_after_hole = np.sum(line_points[r, c, 1] < 0)

        if nr_line_points_below_surface_after_hole > 0:
            return True

        # all points that are above the hole
        r, c = np.where((line_points[:, :, 0] > (self.hole_x - self.hole_width / 2)) & (
                line_points[:, :, 0] < (self.hole_x + self.hole_width / 2)))

        # check if any of those points are below surface
        nr_line_points_below_surface_in_hole = np.sum(line_points[r, c, 1] < -self.hole_depth)

        if nr_line_points_below_surface_in_hole > 0:
            return True

        return False

    def render(self, mode='human'):
        if self.fig is None:
            self.fig = plt.figure()
            # plt.ion()
            # plt.pause(0.01)
        else:
            plt.figure(self.fig.number)

        if mode == "human":
            plt.cla()
            plt.title(f"Iteration: {self._steps}, distance: {self.end_effector - self.bottom_center_of_hole}")

            # Arm
            plt.plot(self._joints[:, 0], self._joints[:, 1], 'ro-', markerfacecolor='k')

            # Add the patch to the Axes
            [plt.gca().add_patch(rect) for rect in self.patches]

            lim = np.sum(self.link_lengths) + 0.5
            plt.xlim([-lim, lim])
            plt.ylim([-1.1, lim])
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

                plt.xlim(-self.n_links, self.n_links), plt.ylim(-1, self.n_links)
                # Arm
                plt.plot(self._joints[:, 0], self._joints[:, 1], 'ro-', markerfacecolor='k')

                plt.pause(0.01)

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)


if __name__ == '__main__':
    nl = 5
    render_mode = "human"  # "human" or "partial" or "final"
    env = HoleReacher(n_links=nl, allow_self_collision=False, allow_wall_collision=False, hole_width=0.15,
                      hole_depth=1, hole_x=1)
    env.reset()
    # env.render(mode=render_mode)

    for i in range(200):
        # objective.load_result("/tmp/cma")
        # test with random actions
        ac = 2 * env.action_space.sample()
        # ac[0] += np.pi/2
        obs, rew, d, info = env.step(ac)
        env.render(mode=render_mode)

        print(rew)

        if d:
            break

    env.close()
