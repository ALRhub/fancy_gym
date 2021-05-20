import cma
from gym import utils
import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import count

from alr_envs.mujoco import alr_mujoco_env
from gym.envs.mujoco import HopperEnv

from t5.algorithm import CMA
from stable_baselines3 import PPO
from stable_baselines3 import SAC


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
    model.learn(total_timesteps=5000, log_interval=4)
    model.save(name)
    # del model # remove to demonstrate saving and loading

def load_sac(env, name):
    model = SAC.load(name)
    obs = env.reset()
    for i in range(10000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()


if __name__ == "__main__":
    #env = gym.make("Hopper-v2")
    env = gym.make("ALRHopper-v0")

    # print(env.unwrapped.sim.model.get_joint_qpos_addr('<body name>'))
    # print(env.unwrapped.sim.model.get_joint_qpos_addr('foot_joint'))

    savename = "sac_hopper"
    loadname = "sac_hopper"

    train_sac(env, savename)
    load_sac(env, loadname)





