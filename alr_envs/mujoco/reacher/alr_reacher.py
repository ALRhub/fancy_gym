import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from alr_envs.utils.utils import angle_normalize


class ALRReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, steps_before_reward=200, n_links=5, balance=False):
        utils.EzPickle.__init__(**locals())

        self._steps = 0
        self.steps_before_reward = steps_before_reward
        self.n_links = n_links

        self.balance = balance
        self.balance_weight = 1.0

        self.reward_weight = 1
        if steps_before_reward == 200:
            self.reward_weight = 200
        elif steps_before_reward == 50:
            self.reward_weight = 50

        if n_links == 5:
            file_name = 'reacher_5links.xml'
        elif n_links == 7:
            file_name = 'reacher_7links.xml'
        else:
            raise ValueError(f"Invalid number of links {n_links}, only 5 or 7 allowed.")

        mujoco_env.MujocoEnv.__init__(self, os.path.join(os.path.dirname(__file__), "assets", file_name), 2)

    def step(self, a):
        self._steps += 1

        reward_dist = 0.0
        angular_vel = 0.0
        reward_balance = 0.0
        if self._steps >= self.steps_before_reward:
            vec = self.get_body_com("fingertip") - self.get_body_com("target")
            reward_dist -= self.reward_weight * np.linalg.norm(vec)
            angular_vel -= np.linalg.norm(self.sim.data.qvel.flat[:self.n_links])
        reward_ctrl = - np.square(a).sum()

        if self.balance:
            reward_balance -= self.balance_weight * np.abs(
                angle_normalize(np.sum(self.sim.data.qpos.flat[:self.n_links]), type="rad"))

        reward = reward_dist + reward_ctrl + angular_vel + reward_balance
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl,
                                      velocity=angular_vel, reward_balance=reward_balance,
                                      end_effector=self.get_body_com("fingertip").copy(),
                                      goal=self.goal if hasattr(self, "goal") else None)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-self.n_links / 10, high=self.n_links / 10, size=2)
            if np.linalg.norm(self.goal) < self.n_links / 10:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        self._steps = 0

        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:self.n_links]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[self.n_links:],  # this is goal position
            self.sim.data.qvel.flat[:self.n_links],  # this is angular velocity
            self.get_body_com("fingertip") - self.get_body_com("target"),
            # self.get_body_com("target"),  # only return target to make problem harder
            [self._steps],
        ])
