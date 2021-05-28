import cma
from gym import utils
import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import count

from alr_envs.mujoco import alr_mujoco_env
from gym.envs.mujoco import HopperEnv

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
    model.learn(total_timesteps=10000, log_interval=4)
    model.save(name)
    # del model # remove to demonstrate saving and loading

def load_sac(env, name):
    model = SAC.load(name)
    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()


def example_sac(modelname):
    env = gym.make("ALRHopper-v0")

    train_sac(env, modelname)
    load_sac(env, modelname)

def example_dmp():
    env = gym.make("alr_envs:ALRHopperEpisodicDMP-v0")
    rewards = 0
    # env.render(mode=None)
    obs = env.reset()

    # number of samples/full trajectories (multiple environment steps)
    for i in range(10):
        obs, reward, done, info = env.step(env.action_space.sample())
        rewards += reward

        if i % 1 == 0:
            # render full DMP trajectory
            # render can only be called once in the beginning as well. That would render every trajectory
            # Calling it after every trajectory allows to modify the mode. mode=None, disables rendering.
            env.render(mode="human")

        if done:
            print(rewards)
            rewards = 0
            obs = env.reset()

if __name__ == "__main__":
    # example_dmp()

    #------------------------------

    # example_sac("sac_hopper_test")

    #-------------------------------

    env = gym.make("ALRHopperEpisodic-v0")
    savename = "episodic_hopper_test"
    loadname = "episodic_hopper_test"

    train_sac(env, savename)
    load_sac(env, loadname)



