import alr_envs


def example_mp(env_name, seed=1):
    """
    Example for running a motion primitive based version of a OpenAI-gym environment, which is already registered.
    For more information on motion primitive specific stuff, look at the mp examples.
    Args:
        env_name: ProMP env_id
        seed: seed

    Returns:

    """
    # While in this case gym.make() is possible to use as well, we recommend our custom make env function.
    env = alr_envs.make(env_name, seed)

    rewards = 0
    obs = env.reset()

    # number of samples/full trajectories (multiple environment steps)
    for i in range(10):
        ac = env.action_space.sample()
        obs, reward, done, info = env.step(ac)
        rewards += reward

        if done:
            print(rewards)
            rewards = 0
            obs = env.reset()


if __name__ == '__main__':
    # DMP - not supported yet
    # example_mp("ReacherDMP-v2")

    # DetProMP
    example_mp("ContinuousMountainCarProMP-v0")
    example_mp("ReacherProMP-v2")
    example_mp("FetchReachDenseProMP-v1")
    example_mp("FetchSlideDenseProMP-v1")
