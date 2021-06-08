import cma
from gym import utils
import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import count

from alr_envs.mujoco import alr_mujoco_env
from gym.envs.mujoco import HopperEnv
from alr_envs.utils.mp_env_async_sampler import AlrMpEnvSampler

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
    for i in range(200):
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

def learn_and_start_dmp():
    # env = gym.make("alr_envs:ALRHopperEpisodicDMP-v0")
    # rewards = 0
    # obs = env.reset()
    
    n = 21  # problem dim, here number of parameters of weight matrix for movement primitive
    n_samples = 14  # how many samples per iteration
    n_cpu = 4  # how many samples to generate in parallel

    env = AlrMpEnvSampler("alr_envs:ALRHopperEpisodicDetPMP-v0", num_envs=n_cpu)

    init_sigma = 1
    x_start = 0.1 * np.random.randn(n, 1)

    # create an instance of the CMA-ES algorithm
    algo = cma.CMAEvolutionStrategy(x0=x_start, sigma0=init_sigma, inopts={"popsize": n_samples})

    t = 0
    max_iters = 100

    opts = []
    while t < max_iters:
        print("----------iter {} -----------".format(t))

        # sample parameters to test
        solutions = algo.ask()
        # collect rollouts with parameters
        obs, new_rewards, done, infos = env(np.vstack(solutions))
        # update search distributioon
        algo.tell(solutions, -new_rewards)  # need to negate rewards as CMA-ES minimizes

        _, opt, __, ___ = env(algo.mean)

        opts.append(opt)

        print(opt)

        t += 1

    print(algo.mean)

    # run learned policy
    test_env = gym.make("alr_envs:ALRHopperEpisodicDetPMP-v0")
    test_env.render('human')
    test_env.reset()
    test_env.step(algo.mean)

if __name__ == "__main__":
    # example_dmp()
    learn_and_start_dmp()

    #------------------------------

    # example_sac("sac_hopper_test")

    #-------------------------------

    # env = gym.make("ALRHopperEpisodic-v0")
    # savename = "episodic_hopper_test"
    # loadname = "episodic_hopper_test"

    # train_sac(env, savename)
    # load_sac(env, loadname)



