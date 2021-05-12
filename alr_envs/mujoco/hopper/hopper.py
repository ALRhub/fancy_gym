from gym import utils
import gym
import numpy as np
from alr_envs.mujoco import alr_mujoco_env

class hopper(alr_mujoco_env.AlrMujocoEnv):
    def __init__(self):
        alr_mujoco_env.AlrMujocoEnv.__init__(self)


def example_hopper():
    env = gym.make('Hopper-v2')
    rewards = 0
    obs = env.reset()

    # number of environment steps
    for i in range(10000):
        obs, reward, done, info = env.step(env.action_space.sample())
        rewards += reward

        if i % 1 == 0:
            env.render()

        if done:
            print(rewards)
            rewards = 0
            obs = env.reset()


if __name__ == "__main__":
    # env = ALRBallInACupEnv()
    # ctxt = np.array([-0.20869846, -0.66376693, 1.18088501])

    # env.configure(ctxt)
    # env.reset()
    # # env.render()
    # for i in range(16000):
    #     # test with random actions
    #     ac = 0.001 * env.action_space.sample()[0:7]
    #     # ac = env.start_pos
    #     # ac[0] += np.pi/2
    #     obs, rew, d, info = env.step(ac)
    #     # env.render()

    #     print(rew)

    #     if d:
    #         break

    # env.close()
    example_hopper()