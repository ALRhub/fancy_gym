import time

import gym

if __name__ == '__main__':

    # env = gym.make('alr_envs:ALRReacher-v0')
    env = gym.make('alr_envs:SimpleReacher-v0')
    # env = gym.make('alr_envs:ALRReacher7-v0')
    state = env.reset()

    for i in range(10000):
        state, reward, done, info = env.step(env.action_space.sample())
        if i % 1 == 0:
            env.render()

        # if done:
        state = env.reset()

        time.sleep(0.5)
