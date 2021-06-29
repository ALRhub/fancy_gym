from alr_envs.dmc.Ball_in_the_cup_mp_wrapper import DMCBallInCupMPWrapper
from alr_envs.utils.make_env_helpers import make_dmp_env, make_env


def example_dmc(env_name="fish-swim", seed=1, iterations=1000):
    env = make_env(env_name, seed)
    rewards = 0
    obs = env.reset()
    print(obs)

    # number of samples(multiple environment steps)
    for i in range(10):
        ac = env.action_space.sample()
        obs, reward, done, info = env.step(ac)
        rewards += reward

        if done:
            print(rewards)
            rewards = 0
            obs = env.reset()


def example_custom_dmc_and_mp(seed=1):
    """
    Example for running a custom motion primitive based environments based off of a dmc task.
    Our already registered environments follow the same structure, but do not directly allow for modifications.
    Hence, this also allows to adjust hyperparameters of the motion primitives more easily.
    We appreciate PRs for custom environments (especially MP wrappers of existing tasks)
    for our repo: https://github.com/ALRhub/alr_envs/
    Args:
        seed: seed

    Returns:

    """

    base_env = "ball_in_cup-catch"
    # Replace this wrapper with the custom wrapper for your environment by inheriting from the MPEnvWrapper.
    # You can also add other gym.Wrappers in case they are needed.
    # wrappers = [HoleReacherMPWrapper]
    wrappers = [DMCBallInCupMPWrapper]
    mp_kwargs = {
        "num_dof": 2,  # env.start_pos
        "num_basis": 5,
        "duration": 2,
        "learn_goal": True,
        "alpha_phase": 2,
        "bandwidth_factor": 2,
        "policy_type": "velocity",
        "weights_scale": 50,
        "goal_scale": 0.1
    }
    env = make_dmp_env(base_env, wrappers=wrappers, seed=seed, **mp_kwargs)
    # OR for a deterministic ProMP:
    # env = make_detpmp_env(base_env, wrappers=wrappers, seed=seed, **mp_args)

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
    # Disclaimer: DMC environments require the seed to be specified in the beginning.
    # Adjusting it afterwards with env.seed() is not recommended as it does not affect the underlying physics.

    # Standard DMC task
    example_dmc("fish_swim", seed=10, iterations=1000)

    # Gym + DMC hybrid task provided in the MP framework
    example_dmc("dmc_ball_in_cup_dmp-v0", seed=10, iterations=10)

    # Custom DMC task
    example_custom_dmc_and_mp()
