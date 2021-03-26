import gym


def example_mujoco():
    env = gym.make('alr_envs:ALRReacher-v0')
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


def example_dmp():
    # env = gym.make("alr_envs:ViaPointReacherDMP-v0")
    env = gym.make("alr_envs:HoleReacherDMP-v0")
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
            env.render(mode="partial")

        if done:
            print(rewards)
            rewards = 0
            obs = env.reset()


if __name__ == '__main__':
    example_dmp()
