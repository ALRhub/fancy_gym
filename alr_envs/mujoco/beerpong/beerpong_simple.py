from gym import utils
import os
import numpy as np
from alr_envs.mujoco import alr_mujoco_env


class ALRBeerpongEnv(alr_mujoco_env.AlrMujocoEnv, utils.EzPickle):
    def __init__(self, n_substeps=4, apply_gravity_comp=True, reward_function=None):
        self._steps = 0

        self.xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets",
                                     "beerpong" + ".xml")

        self.start_pos = np.array([0.0, 1.35, 0.0, 1.18, 0.0, -0.786, -1.59])
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
                                             apply_gravity_comp=apply_gravity_comp,
                                             n_substeps=n_substeps)

        self.sim_time = 8  # seconds
        self.sim_steps = int(self.sim_time / self.dt)
        if reward_function is None:
            from alr_envs.mujoco.beerpong.beerpong_reward_simple import BeerpongReward
            reward_function = BeerpongReward
        self.reward_function = reward_function(self.sim, self.sim_steps)
        self.cup_robot_id = self.sim.model._site_name2id["cup_robot_final"]
        self.ball_id = self.sim.model._body_name2id["ball"]
        self.cup_table_id = self.sim.model._body_name2id["cup_table"]
        # self.bounce_table_id = self.sim.model._body_name2id["bounce_table"]

    @property
    def current_pos(self):
        return self.sim.data.qpos[0:7].copy()

    @property
    def current_vel(self):
        return self.sim.data.qvel[0:7].copy()

    def configure(self, context):
        if context is None:
            context = np.array([0, -2, 0.840])
        self.context = context
        self.reward_function.reset(context)

    def reset_model(self):
        init_pos_all = self.init_qpos.copy()
        init_pos_robot = self.start_pos
        init_vel = np.zeros_like(init_pos_all)

        self._steps = 0
        self._q_pos = []
        self._q_vel = []

        start_pos = init_pos_all
        start_pos[0:7] = init_pos_robot
        # start_pos[7:] = np.copy(self.sim.data.site_xpos[self.cup_robot_id, :]) + np.array([0., 0.0, 0.05])

        self.set_state(start_pos, init_vel)

        ball_pos = np.copy(self.sim.data.site_xpos[self.cup_robot_id, :]) + np.array([0., 0.0, 0.05])
        self.sim.model.body_pos[self.ball_id] = ball_pos.copy()
        self.sim.model.body_pos[self.cup_table_id] = self.context.copy()
        # self.sim.model.body_pos[self.bounce_table_id] = self.context.copy()

        self.sim.forward()

        return self._get_obs()

    def step(self, a):
        reward_dist = 0.0
        angular_vel = 0.0
        reward_ctrl = - np.square(a).sum()
        action_cost = np.sum(np.square(a))

        crash = self.do_simulation(a)
        joint_cons_viol = self.check_traj_in_joint_limits()

        self._q_pos.append(self.sim.data.qpos[0:7].ravel().copy())
        self._q_vel.append(self.sim.data.qvel[0:7].ravel().copy())

        ob = self._get_obs()

        if not crash and not joint_cons_viol:
            reward, success, stop_sim = self.reward_function.compute_reward(a, self.sim, self._steps)
            done = success or self._steps == self.sim_steps - 1 or stop_sim
            self._steps += 1
        else:
            reward = -10 - 1e-2 * action_cost
            success = False
            done = True
        return ob, reward, done, dict(reward_dist=reward_dist,
                                      reward_ctrl=reward_ctrl,
                                      velocity=angular_vel,
                                      traj=self._q_pos, is_success=success,
                                      is_collided=crash or joint_cons_viol)

    def check_traj_in_joint_limits(self):
        return any(self.current_pos > self.j_max) or any(self.current_pos < self.j_min)

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

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:7]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            # self.get_body_com("target"),  # only return target to make problem harder
            [self._steps],
        ])



if __name__ == "__main__":
    env = ALRBeerpongEnv()
    ctxt = np.array([0, -2, 0.840])    # initial

    env.configure(ctxt)
    env.reset()
    env.render()
    for i in range(16000):
        # test with random actions
        ac = 0.0 * env.action_space.sample()[0:7]
        ac[1] = -0.01
        ac[3] = - 0.01
        ac[5] = -0.01
        # ac = env.start_pos
        # ac[0] += np.pi/2
        obs, rew, d, info = env.step(ac)
        env.render()

        print(rew)

        if d:
            break

    env.close()

