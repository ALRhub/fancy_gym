from gym import utils
import os
import numpy as np
from alr_envs.mujoco import alr_mujoco_env


class ALRBallInACupEnv(alr_mujoco_env.AlrMujocoEnv, utils.EzPickle):
    def __init__(self, n_substeps=4, apply_gravity_comp=True, simplified: bool = False,
                 reward_type: str = None, context: np.ndarray = None):
        self._steps = 0

        self.xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets",
                                     "biac_base" + ".xml")

        self._q_pos = []
        self._q_vel = []
        # self.weight_matrix_scale = 50
        self.max_ctrl = np.array([150., 125., 40., 60., 5., 5., 2.])
        self.p_gains = 1 / self.max_ctrl * np.array([200, 300, 100, 100, 10, 10, 2.5])
        self.d_gains = 1 / self.max_ctrl * np.array([7, 15, 5, 2.5, 0.3, 0.3, 0.05])

        self.j_min = np.array([-2.6, -1.985, -2.8, -0.9, -4.55, -1.5707, -2.7])
        self.j_max = np.array([2.6, 1.985, 2.8, 3.14159, 1.25, 1.5707, 2.7])

        self.context = context

        utils.EzPickle.__init__(self)
        alr_mujoco_env.AlrMujocoEnv.__init__(self,
                                             self.xml_path,
                                             apply_gravity_comp=apply_gravity_comp,
                                             n_substeps=n_substeps)
        self._start_pos = np.array([0.0, 0.58760536, 0.0, 1.36004913, 0.0, -0.32072943, -1.57])
        self._start_vel = np.zeros(7)

        self.simplified = simplified

        self.sim_time = 8  # seconds
        self.sim_steps = int(self.sim_time / self.dt)
        if reward_type == "no_context":
            from alr_envs.mujoco.ball_in_a_cup.ball_in_a_cup_reward_simple import BallInACupReward
            reward_function = BallInACupReward
        elif reward_type == "contextual_goal":
            from alr_envs.mujoco.ball_in_a_cup.ball_in_a_cup_reward import BallInACupReward
            reward_function = BallInACupReward
        else:
            raise ValueError("Unknown reward type: {}".format(reward_type))
        self.reward_function = reward_function(self.sim_steps)

    @property
    def start_pos(self):
        if self.simplified:
            return self._start_pos[1::2]
        else:
            return self._start_pos

    @property
    def start_vel(self):
        if self.simplified:
            return self._start_vel[1::2]
        else:
            return self._start_vel

    @property
    def current_pos(self):
        return self.sim.data.qpos[0:7].copy()

    @property
    def current_vel(self):
        return self.sim.data.qvel[0:7].copy()

    def reset(self):
        self.reward_function.reset(None)
        return super().reset()

    def reset_model(self):
        init_pos_all = self.init_qpos.copy()
        init_pos_robot = self._start_pos
        init_vel = np.zeros_like(init_pos_all)

        self._steps = 0
        self._q_pos = []
        self._q_vel = []

        start_pos = init_pos_all
        start_pos[0:7] = init_pos_robot

        self.set_state(start_pos, init_vel)

        return self._get_obs()

    def step(self, a):
        reward_dist = 0.0
        angular_vel = 0.0
        reward_ctrl = - np.square(a).sum()

        crash = self.do_simulation(a)
        # joint_cons_viol = self.check_traj_in_joint_limits()

        self._q_pos.append(self.sim.data.qpos[0:7].ravel().copy())
        self._q_vel.append(self.sim.data.qvel[0:7].ravel().copy())

        ob = self._get_obs()

        if not crash:
            reward, success, is_collided = self.reward_function.compute_reward(a, self)
            done = success or self._steps == self.sim_steps - 1 or is_collided
            self._steps += 1
        else:
            reward = -2000
            success = False
            is_collided = False
            done = True
        return ob, reward, done, dict(reward_dist=reward_dist,
                                      reward_ctrl=reward_ctrl,
                                      velocity=angular_vel,
                                      traj=self._q_pos, is_success=success,
                                      is_collided=is_collided, sim_crash=crash)

    def check_traj_in_joint_limits(self):
        return any(self.current_pos > self.j_max) or any(self.current_pos < self.j_min)

    # TODO: extend observation space
    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:7]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            # self.get_body_com("target"),  # only return target to make problem harder
            [self._steps],
        ])

    # TODO
    @property
    def active_obs(self):
        return np.hstack([
            [False] * 7,  # cos
            [False] * 7,  # sin
            # [True] * 2,  # x-y coordinates of target distance
            [False]  # env steps
        ])

    # These functions are for the task with 3 joint actuations
    def extend_des_pos(self, des_pos):
        des_pos_full = self._start_pos.copy()
        des_pos_full[1] = des_pos[0]
        des_pos_full[3] = des_pos[1]
        des_pos_full[5] = des_pos[2]
        return des_pos_full

    def extend_des_vel(self, des_vel):
        des_vel_full = self._start_vel.copy()
        des_vel_full[1] = des_vel[0]
        des_vel_full[3] = des_vel[1]
        des_vel_full[5] = des_vel[2]
        return des_vel_full

    def render(self, render_mode, **render_kwargs):
        if render_mode == "plot_trajectory":
            if self._steps == 1:
                import matplotlib.pyplot as plt
                # plt.ion()
                self.fig, self.axs = plt.subplots(3, 1)

            if self._steps <= 1750:
                for ax, cp in zip(self.axs, self.current_pos[1::2]):
                    ax.scatter(self._steps, cp, s=2, marker=".")

            # self.fig.show()

        else:
            super().render(render_mode, **render_kwargs)


if __name__ == "__main__":
    env = ALRBallInACupEnv()
    ctxt = np.array([-0.20869846, -0.66376693, 1.18088501])

    env.configure(ctxt)
    env.reset()
    # env.render()
    for i in range(16000):
        # test with random actions
        ac = 0.001 * env.action_space.sample()[0:7]
        # ac = env.start_pos
        # ac[0] += np.pi/2
        obs, rew, d, info = env.step(ac)
        # env.render()

        print(rew)

        if d:
            break

    env.close()

