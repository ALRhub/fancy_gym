import numpy as np

import fancy_gym


def test_env(env_id="3dof-hit", seed=0, iteration=5):
    env = fancy_gym.make(env_id=env_id, seed=seed)

    for i in range(iteration):
        obs = env.reset()
        stp = 0
        while True:
            act = env.action_space.sample()
            obs_, reward, done, info = env.step(act)
            ra = env.render("rgb_array")
            print(ra.shape)
            stp += 1
            if done:
                break


def test_mp_env(env_id="3dof-ProMP-hit", seed=0, iteration=5):
    env = fancy_gym.make(env_id=env_id, seed=seed)

    for i in range(iteration):
        obs = env.reset()
        if i == 0:
            env.render(mode="human")
        while True:
            act = env.action_space.sample()
            obs, reward, done, info = env.step(act)
            if done:
                break


if __name__ == "__main__":
    test_mp_env(env_id="3dof-ProMP-hit", seed=0, iteration=10)
