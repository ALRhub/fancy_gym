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

    def configure(self, context):
        self.context = context
        self.reward_function.reset(context)

    def reset_model(self):
        start_pos = self.init_qpos.copy()
        start_pos[0:7] = self.start_pos
        start_vel = np.zeros_like(start_pos)
        self.set_state(start_pos, start_vel)
        self._steps = 0
        self._q_pos = []

    def step(self, a):
        # Apply gravity compensation
        if not np.all(self.sim.data.qfrc_applied[:7] == self.sim.data.qfrc_bias[:7]):
            self.sim.data.qfrc_applied[:7] = self.sim.data.qfrc_bias[:7]

        reward_dist = 0.0
        angular_vel = 0.0
        # if self._steps >= self.steps_before_reward:
        #     vec = self.get_body_com("fingertip") - self.get_body_com("target")
        #     reward_dist -= self.reward_weight * np.linalg.norm(vec)
        #     angular_vel -= np.linalg.norm(self.sim.data.qvel.flat[:self.n_links])
        reward_ctrl = - np.square(a).sum()
        # reward_balance = - self.balance_weight * np.abs(
        #     angle_normalize(np.sum(self.sim.data.qpos.flat[:self.n_links]), type="rad"))
        #
        # reward = reward_dist + reward_ctrl + angular_vel + reward_balance
        # self.do_simulation(a, self.frame_skip)

        crash = self.do_simulation(a, self.frame_skip)

        self._q_pos.append(self.sim.data.qpos[0:7].ravel().copy())

        ob = self._get_obs()

        if not crash:
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
    env.configure(None)
    env.reset()
    for i in range(2000):
        # objective.load_result("/tmp/cma")
        # test with random actions
        # ac = 0.0 * env.action_space.sample()
        ac = env.start_pos
        # ac[0] += np.pi/2
        obs, rew, d, info = env.step(ac)
        env.render()

        print(rew)

        if d:
            break

    env.close()

