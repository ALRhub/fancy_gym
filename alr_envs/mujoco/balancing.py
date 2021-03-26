import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from alr_envs.utils.utils import angle_normalize


class BalancingEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, n_links=5):
        utils.EzPickle.__init__(**locals())

        self.n_links = n_links

        if n_links == 5:
            file_name = 'reacher_5links.xml'
        elif n_links == 7:
            file_name = 'reacher_7links.xml'
        else:
            raise ValueError(f"Invalid number of links {n_links}, only 5 or 7 allowed.")

        mujoco_env.MujocoEnv.__init__(self, os.path.join(os.path.dirname(__file__), "assets", file_name), 2)

    def step(self, a):
        angle = angle_normalize(np.sum(self.sim.data.qpos.flat[:self.n_links]), type="rad")
        reward = - np.abs(angle)

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(angle=angle, end_effector=self.get_body_com("fingertip").copy())

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1

    def reset_model(self):
        # This also generates a goal, we however do not need/use it
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        qpos[-2:] = 0
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)

        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:self.n_links]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qvel.flat[:self.n_links],  # this is angular velocity
        ])
