from alr_envs.mujoco import alr_mujoco_env
from gym import utils, spaces
import os
import numpy as np
from alr_envs.mujoco.ball_in_a_cup.ball_in_a_cup_reward import BallInACupReward


class ALRBallInACupEnv(alr_mujoco_env.AlrMujocoEnv, utils.EzPickle):
    def __init__(self, reward_function=None):
        self._steps = 0

        self.xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets",
                                     "biac_base" + ".xml")

        self.sim_time = 8  # seconds
        self.sim_steps = int(self.sim_time / (0.0005 * 4))  # circular dependency.. sim.dt <-> mujocoenv init <-> reward fct
        self.reward_function = reward_function(self.sim_steps)

        self.start_pos = np.array([0.0, 0.58760536, 0.0, 1.36004913, 0.0, -0.32072943, -1.57])
        self.start_vel = np.zeros(7)

        self._q_pos = []
        self._q_vel = []
        # self.weight_matrix_scale = 50
        self.max_ctrl = np.array([150., 125., 40., 60., 5., 5., 2.])
        self.p_gains = 1 / self.max_ctrl * np.array([200, 300, 100, 100, 10, 10, 2.5])
        self.d_gains = 1 / self.max_ctrl * np.array([7, 15, 5, 2.5, 0.3, 0.3, 0.05])

        self.j_min = np.array([-2.6, -1.985, -2.8, -0.9, -4.55, -1.5707, -2.7])
        self.j_max = np.array([2.6, 1.985, 2.8, 3.14159, 1.25, 1.5707, 2.7])

        self.context = None

        utils.EzPickle.__init__(self)
        alr_mujoco_env.AlrMujocoEnv.__init__(self,
                                             self.xml_path,
                                             apply_gravity_comp=True,
                                             n_substeps=4)

    @property
    def current_pos(self):
        return self.sim.data.qpos[0:7].copy()

    @property
    def current_vel(self):
        return self.sim.data.qvel[0:7].copy()

    def configure(self, context):
        self.context = context

    def reset_model(self):
        init_pos_all = self.init_qpos.copy()
        init_pos_robot = self.start_pos
        init_vel = np.zeros_like(init_pos_all)
        ball_id = self.sim.model._body_name2id["ball"]
        goal_final_id = self.sim.model._site_name2id["cup_goal_final"]

        self._steps = 0
        self.reward_function.reset()
        self._q_pos = []
        self._q_vel = []

        start_pos = init_pos_all
        start_pos[0:7] = init_pos_robot

        self.set_state(start_pos, init_vel)

        # Reset the system
        # self.sim.data.qpos[:] = init_pos_all
        # self.sim.data.qvel[:] = init_vel
        # self.sim.data.qpos[0:7] = init_pos_robot
        #
        # self.sim.step()
        #
        # self.sim.data.qpos[:] = init_pos_all
        # self.sim.data.qvel[:] = init_vel
        # self.sim.data.qpos[0:7] = init_pos_robot
        # self.sim.data.body_xpos[ball_id, :] = np.copy(self.sim.data.site_xpos[goal_final_id, :]) - np.array([0., 0., 0.329])
        #
        # # Stabilize the system around the initial position
        # for i in range(0, 500):
        #     self.sim.data.qpos[7:] = 0.
        #     self.sim.data.qvel[7:] = 0.
        #     # self.sim.data.qpos[7] = -0.2
        #     cur_pos = self.sim.data.qpos[0:7].copy()
        #     cur_vel = self.sim.data.qvel[0:7].copy()
        #     trq = self.p_gains * (init_pos_robot - cur_pos) + self.d_gains * (np.zeros_like(init_pos_robot) - cur_vel)
        #     self.sim.data.qfrc_applied[0:7] = trq + self.sim.data.qfrc_bias[:7].copy()
        #     self.sim.step()
        #     self.render()
        #
        # for i in range(0, 500):
        #     cur_pos = self.sim.data.qpos[0:7].copy()
        #     cur_vel = self.sim.data.qvel[0:7].copy()
        #     trq = self.p_gains * (init_pos_robot - cur_pos) + self.d_gains * (np.zeros_like(init_pos_robot) - cur_vel)
        #     self.sim.data.qfrc_applied[0:7] = trq + self.sim.data.qfrc_bias[:7].copy()
        #     self.sim.step()
        #     self.render()

        return self._get_obs()

    def step(self, a):
        reward_dist = 0.0
        angular_vel = 0.0
        reward_ctrl = - np.square(a).sum()

        crash = self.do_simulation(a)
        joint_cons_viol = self.check_traj_in_joint_limits()

        self._q_pos.append(self.sim.data.qpos[0:7].ravel().copy())
        self._q_vel.append(self.sim.data.qvel[0:7].ravel().copy())

        ob = self._get_obs()

        if not crash and not joint_cons_viol:
            reward, success, collision = self.reward_function.compute_reward(a, self.sim, self._steps)
            done = success or self._steps == self.sim_steps - 1 or collision
            self._steps += 1
        else:
            reward = -1000
            done = True
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl,
                                      velocity=angular_vel, # reward_balance=reward_balance,
                                      # end_effector=self.get_body_com("fingertip").copy(),
                                      goal=self.goal if hasattr(self, "goal") else None,
                                      traj=self._q_pos)

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

    def extend_des_pos(self, des_pos):
        des_pos_full = self.start_pos.copy()
        des_pos_full[1] = des_pos[0]
        des_pos_full[3] = des_pos[1]
        des_pos_full[5] = des_pos[2]
        return des_pos_full

    def extend_des_vel(self, des_vel):
        des_vel_full = self.start_vel.copy()
        des_vel_full[1] = des_vel[0]
        des_vel_full[3] = des_vel[1]
        des_vel_full[5] = des_vel[2]
        return des_vel_full


if __name__ == "__main__":
    env = ALRBallInACupEnv(reward_function=BallInACupReward)
    env.reset()
    env.render()
    for i in range(4000):
        # objective.load_result("/tmp/cma")
        # test with random actions
        # ac = 0.1 * env.action_space.sample()
        # ac = -np.array([i, i, i]) / 10000 + np.array([env.start_pos[1], env.start_pos[3], env.start_pos[5]])
        # ac = np.array([0., -10, 0, -1, 0, 1, 0])
        ac = np.array([0.,0., 0, 0, 0, 0, 0])
        # ac[0] += np.pi/2
        obs, rew, d, info = env.step(ac)
        env.render()

        print(rew)

        if d:
            break

    env.close()

