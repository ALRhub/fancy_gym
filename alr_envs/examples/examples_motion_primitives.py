import alr_envs


def example_mp(env_name="alr_envs:HoleReacherDMP-v1", seed=1, iterations=1, render=True):
    """
    Example for running a motion primitive based environment, which is already registered
    Args:
        env_name: DMP env_id
        seed: seed for deterministic behaviour
        iterations: Number of rollout steps to run
        render: Render the episode

    Returns:

    """
    # While in this case gym.make() is possible to use as well, we recommend our custom make env function.
    # First, it already takes care of seeding and second enables the use of DMC tasks within the gym interface.
    env = alr_envs.make(env_name, seed)

    rewards = 0
    # env.render(mode=None)
    obs = env.reset()

    # number of samples/full trajectories (multiple environment steps)
    for i in range(iterations):

        if render and i % 2 == 0:
            # This renders the full MP trajectory
            # It is only required to call render() once in the beginning, which renders every consecutive trajectory.
            # Resetting to no rendering, can be achieved by render(mode=None).
            # It is also possible to change the mode multiple times when
            # e.g. only every second trajectory should be displayed, such as here
            # Just make sure the correct mode is set before executing the step.
            env.render(mode="human")
        else:
            env.render(mode=None)

        ac = env.action_space.sample()
        obs, reward, done, info = env.step(ac)
        rewards += reward

        if done:
            print(rewards)
            rewards = 0
            obs = env.reset()


def example_custom_mp(env_name="alr_envs:HoleReacherDMP-v1", seed=1, iterations=1, render=True):
    """
    Example for running a motion primitive based environment, which is already registered
    Args:
        env_name: DMP env_id
        seed: seed for deterministic behaviour
        iterations: Number of rollout steps to run
        render: Render the episode

    Returns:

    """
    # Changing the mp_kwargs is possible by providing them to gym.
    # E.g. here by providing way to many basis functions
    mp_kwargs = {
        "num_dof": 5,
        "num_basis": 1000,
        "duration": 2,
        "learn_goal": True,
        "alpha_phase": 2,
        "bandwidth_factor": 2,
        "policy_type": "velocity",
        "weights_scale": 50,
        "goal_scale": 0.1
    }
    env = alr_envs.make(env_name, seed, mp_kwargs=mp_kwargs)

    # This time rendering every trajectory
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
            print(rewards)
            rewards = 0
            obs = env.reset()


def example_fully_custom_mp(seed=1, iterations=1, render=True):
    """
    Example for running a custom motion primitive based environments.
    Our already registered environments follow the same structure.
    Hence, this also allows to adjust hyperparameters of the motion primitives.
    Yet, we recommend the method above if you are just interested in chaining those parameters for existing tasks.
    We appreciate PRs for custom environments (especially MP wrappers of existing tasks) 
    for our repo: https://github.com/ALRhub/alr_envs/
    Args:
        seed: seed
        iterations: Number of rollout steps to run
        render: Render the episode

    Returns:

    """

    base_env = "alr_envs:HoleReacher-v1"

    # Replace this wrapper with the custom wrapper for your environment by inheriting from the MPEnvWrapper.
    # You can also add other gym.Wrappers in case they are needed.
    wrappers = [alr_envs.alr.classic_control.hole_reacher.MPWrapper]
    mp_kwargs = {
        "num_dof": 5,
        "num_basis": 5,
        "duration": 2,
        "learn_goal": True,
        "alpha_phase": 2,
        "bandwidth_factor": 2,
        "policy_type": "velocity",
        "weights_scale": 50,
        "goal_scale": 0.1
    }
    env = alr_envs.make_dmp_env(base_env, wrappers=wrappers, seed=seed, mp_kwargs=mp_kwargs)
    # OR for a deterministic ProMP:
    # env = make_detpmp_env(base_env, wrappers=wrappers, seed=seed, mp_kwargs=mp_kwargs)

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
            print(rewards)
            rewards = 0
            obs = env.reset()


if __name__ == '__main__':
    render = True
    # DMP
    example_mp("alr_envs:HoleReacherDMP-v1", seed=10, iterations=1, render=render)

    # ProMP
    example_mp("alr_envs:HoleReacherProMP-v1", seed=10, iterations=1, render=render)

    # DetProMP
    example_mp("alr_envs:HoleReacherDetPMP-v1", seed=10, iterations=1, render=render)

    # Altered basis functions
    example_custom_mp("alr_envs:HoleReacherDMP-v1", seed=10, iterations=1, render=render)

    # Custom MP
    example_fully_custom_mp(seed=10, iterations=1, render=render)
