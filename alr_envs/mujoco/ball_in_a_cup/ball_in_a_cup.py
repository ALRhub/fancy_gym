from gym import utils
import os
import numpy as np
from alr_envs.mujoco import alr_mujoco_env
from alr_envs.mujoco.ball_in_a_cup.ball_in_a_cup_reward import BallInACupReward
import mujoco_py


class ALRBallInACupEnv(alr_mujoco_env.AlrMujocoEnv, utils.EzPickle):
    def __init__(self, n_substeps=4, apply_gravity_comp=True, reward_function=None):
        self._steps = 0

        self.xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets",
                                     "biac_base" + ".xml")

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
                                             apply_gravity_comp=apply_gravity_comp,
                                             n_substeps=n_substeps)

        self.sim_time = 8  # seconds
        self.sim_steps = int(self.sim_time / self.dt)
        if reward_function is None:
            from alr_envs.mujoco.ball_in_a_cup.ball_in_a_cup_reward import BallInACupReward
            reward_function = BallInACupReward
        self.reward_function = reward_function(self.sim_steps)

    @property
    def current_pos(self):
        return self.sim.data.qpos[0:7].copy()

    @property
    def current_vel(self):
        return self.sim.data.qvel[0:7].copy()

    def configure(self, context):
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

        self.set_state(start_pos, init_vel)

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
            reward, success, stop_sim = self.reward_function.compute_reward(a, self.sim, self._steps)
            done = success or self._steps == self.sim_steps - 1 or stop_sim
            self._steps += 1
        else:
            reward = -1000
            success = False
            done = True
        return ob, reward, done, dict(reward_dist=reward_dist,
                                      reward_ctrl=reward_ctrl,
                                      velocity=angular_vel,
                                      traj=self._q_pos, is_success=success,
                                      is_collided=crash or joint_cons_viol)

    def check_traj_in_joint_limits(self):
        return any(self.current_pos > self.j_max) or any(self.current_pos < self.j_min)

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:7]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            # self.get_body_com("target"),  # only return target to make problem harder
            [self._steps],
        ])



if __name__ == "__main__":
    env = ALRBallInACupEnv()
    ctxt = np.array([-0.20869846, -0.66376693, 1.18088501])

    env.configure(ctxt)
    env.reset()
    env.render()
    for i in range(2000):
        # test with random actions
        ac = 0.01 * env.action_space.sample()[0:7]
        # ac = env.start_pos
        # ac[0] += np.pi/2
        obs, rew, d, info = env.step(ac)
        env.render()

        print(rew)

        if d:
            break

    env.close()

