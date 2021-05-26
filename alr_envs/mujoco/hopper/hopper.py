import cma
from gym import utils
import gym
import numpy as np
from alr_envs.mujoco import alr_mujoco_env
from gym.envs.mujoco import HopperEnv

from stable_baselines3 import PPO
from stable_baselines3 import SAC


class ALRHopperEnv(HopperEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()

    def step(self, a):
        height_before = self.sim.data.qpos[1]
        foot_height_before = self.get_body_com("foot")[2]
        self.do_simulation(a, self.frame_skip)
        pos, height, angle = self.sim.data.qpos[0:3]
        foot_height = self.get_body_com("foot")[2]
        # print(foot_height)
        alive_bonus = 1
        reward = (foot_height - foot_height_before) #/ self.dt
        reward = (height - height_before) #/ self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        #print(reward)
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all()
                    and (height > .6))
        # done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
        #              (height > .7) and (abs(angle) < .2))
        obs = self._get_obs()
        return obs, reward, done, {}
