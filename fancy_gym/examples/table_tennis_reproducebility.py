import fancy_gym
import numpy as np

env_1 = fancy_gym.make("TableTennis4DProDMP-v0", seed=0)
env_2 = fancy_gym.make("TableTennis4DProDMP-v0", seed=0)

obs_1 = env_1.reset()
obs_2 = env_2.reset()
assert np.all(obs_1 == obs_2), "The observations should be the same"
for i in range(100000):
    action = env_1.action_space.sample()
    obs_1, reward_1, done_1, info_1 = env_1.step(action)
    obs_2, reward_2, done_2, info_2 = env_2.step(action)
    assert np.all(obs_1 == obs_2), "The observations should be the same"
    assert np.all(reward_1 == reward_2), "The rewards should be the same"
    assert np.all(done_1 == done_2), "The done flags should be the same"
    for key in info_1:
        assert np.all(info_1[key] == info_2[key]), f"The info fields: {key} should be the same"
    if done_1 and done_2:
        obs_1 = env_1.reset()
        obs_2 = env_2.reset()
        assert np.all(obs_1 == obs_2), "The observations should be the same"