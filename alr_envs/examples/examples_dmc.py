import alr_envs
from alr_envs.dmc.suite.ball_in_cup.mp_wrapper import MPWrapper


def example_dmc(env_id="fish-swim", seed=1, iterations=1000, render=True):
    """
    Example for running a DMC based env in the step based setting.
    The env_id has to be specified as `domain_name-task_name` or
    for manipulation tasks as `manipulation-environment_name`

    Args:
        env_id: Either `domain_name-task_name` or `manipulation-environment_name`
        seed: seed for deterministic behaviour
        iterations: Number of rollout steps to run
        render: Render the episode

    Returns:

    """
    env = alr_envs.make_env(env_id, seed)
    rewards = 0
    obs = env.reset()
    print("observation shape:", env.observation_space.shape)
    print("action shape:", env.action_space.shape)

    for i in range(iterations):
        ac = env.action_space.sample()
        obs, reward, done, info = env.step(ac)
        rewards += reward

        if render:
            env.render("human")

        if done:
            print(env_id, rewards)
            rewards = 0
            obs = env.reset()

    env.close()
    del env


def example_custom_dmc_and_mp(seed=1, iterations=1, render=True):
    """
    Example for running a custom motion primitive based environments.
    Our already registered environments follow the same structure.
    Hence, this also allows to adjust hyperparameters of the motion primitives.
    Yet, we recommend the method above if you are just interested in chaining those parameters for existing tasks.
    We appreciate PRs for custom environments (especially MP wrappers of existing tasks)
    for our repo: https://github.com/ALRhub/alr_envs/
    Args:
        seed: seed for deterministic behaviour
        iterations: Number of rollout steps to run
        render: Render the episode

    Returns:

    """

    # Base DMC name, according to structure of above example
    base_env = "ball_in_cup-catch"

    # Replace this wrapper with the custom wrapper for your environment by inheriting from the MPEnvWrapper.
    # You can also add other gym.Wrappers in case they are needed.
    wrappers = [MPWrapper]
    mp_kwargs = {
        "num_dof": 2,
        "num_basis": 5,
        "duration": 20,
        "learn_goal": True,
        "alpha_phase": 2,
        "bandwidth_factor": 2,
        "policy_type": "motor",
        "weights_scale": 50,
        "goal_scale": 0.1,
        "policy_kwargs": {
            "p_gains": 0.2,
            "d_gains": 0.05
        }
    }
    kwargs = {
        "time_limit": 20,
        "episode_length": 1000,
        # "frame_skip": 1
    }
    env = alr_envs.make_dmp_env(base_env, wrappers=wrappers, seed=seed, mp_kwargs=mp_kwargs, **kwargs)
    # OR for a deterministic ProMP:
    # env = alr_envs.make_detpmp_env(base_env, wrappers=wrappers, seed=seed, mp_kwargs=mp_args)

    # This renders the full MP trajectory
    # It is only required to call render() once in the beginning, which renders every consecutive trajectory.
    # Resetting to no rendering, can be achieved by render(mode=None).
    # It is also possible to change them mode multiple times when
    # e.g. only every nth trajectory should be displayed.
    if render:
        env.render(mode="human")

    rewards = 0
    obs = env.reset()

    # number of samples/full trajectories (multiple environment steps)
    for i in range(iterations):
        ac = env.action_space.sample()
        obs, reward, done, info = env.step(ac)
        rewards += reward

        if done:
            print(base_env, rewards)
            rewards = 0
            obs = env.reset()

    env.close()
    del env


if __name__ == '__main__':
    # Disclaimer: DMC environments require the seed to be specified in the beginning.
    # Adjusting it afterwards with env.seed() is not recommended as it does not affect the underlying physics.

    # For rendering DMC
    # export MUJOCO_GL="osmesa"
    render = False

    # # Standard DMC Suite tasks
    example_dmc("fish-swim", seed=10, iterations=1000, render=render)

    # Manipulation tasks
    # Disclaimer: The vision versions are currently not integrated and yield an error
    example_dmc("manipulation-reach_site_features", seed=10, iterations=250, render=render)

    # Gym + DMC hybrid task provided in the MP framework
    example_dmc("dmc_ball_in_cup-catch_detpmp-v0", seed=10, iterations=1, render=render)

    # Custom DMC task
    # Different seed, because the episode is longer for this example and the name+seed combo is already registered above
    example_custom_dmc_and_mp(seed=11, iterations=1, render=render)
