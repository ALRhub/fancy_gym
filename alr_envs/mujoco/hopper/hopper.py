import cma
from gym import utils
import gym
import numpy as np
from alr_envs.mujoco import alr_mujoco_env
from gym.envs.mujoco import HopperEnv
from t5.algorithm import CMA

from stable_baselines3 import PPO
from stable_baselines3 import SAC

class ALRHopperEnv(HopperEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()

    def step(self, a):
        heightbefore = self.sim.data.qpos[1]
        self.do_simulation(a, self.frame_skip)
        pos, height, angle = self.sim.data.qpos[0:3]
        alive_bonus = 0.1
        reward = (height - heightbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all()
                    and (height > .7))
        # done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
        #              (height > .7) and (abs(angle) < .2))
        obs = self._get_obs()
        return obs, reward, done, {}


def example_ppo(env):
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)

    obs = env.reset()
    for i in range(10000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

    env.close()


def train_sac(env, name):
    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000, log_interval=4)
    model.save(name)
    # del model # remove to demonstrate saving and loading


def load_sac(env, name):
    # sac_hopper: Sprünge nach vorne
    model = SAC.load(name)

    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()


if __name__ == "__main__":
    #env = gym.make("Hopper-v2")
    env = gym.make("ALRHopper-v0")

    savename = "sac_hopper3"
    loadname = "sac_hopper2"
    
    #train_sac(env, savename)
    load_sac(env, loadname)
    # example_sac(env)
