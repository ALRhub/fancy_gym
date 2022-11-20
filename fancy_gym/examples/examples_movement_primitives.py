import fancy_gym


def example_mp(env_name="HoleReacherProMP-v0", seed=1, iterations=1, render=True):
    """
    Example for running a black box based environment, which is already registered
    Args:
        env_name: Black box env_id
        seed: seed for deterministic behaviour
        iterations: Number of rollout steps to run
        render: Render the episode

    Returns:

    """
    # Equivalent to gym, we have a make function which can be used to create environments.
    # It takes care of seeding and enables the use of a variety of external environments using the gym interface.
    env = fancy_gym.make(env_name, seed)

    returns = 0
    # env.render(mode=None)
    obs = env.reset()

    # number of samples/full trajectories (multiple environment steps)
    for i in range(iterations):

        if render and i % 1 == 0:
            # This renders the full MP trajectory
            # It is only required to call render() once in the beginning, which renders every consecutive trajectory.
            # Resetting to no rendering, can be achieved by render(mode=None).
            # It is also possible to change the mode multiple times when
            # e.g. only every second trajectory should be displayed, such as here
            # Just make sure the correct mode is set before executing the step.
            env.render(mode="human")
        else:
            env.render(mode=None)

        # Now the action space is not the raw action but the parametrization of the trajectory generator,
        # such as a ProMP
        ac = env.action_space.sample()
        # This executes a full trajectory and gives back the context (obs) of the last step in the trajectory, or the
        # full observation space of the last step, if replanning/sub-trajectory learning is used. The 'reward' is equal
        # to the return of a trajectory. Default is the sum over the step-wise rewards.
        obs, reward, done, info = env.step(ac)
        # Aggregated returns
        returns += reward

        if done:
            print(reward)
            obs = env.reset()


def example_custom_mp(env_name="Reacher5dProMP-v0", seed=1, iterations=1, render=True):
    """
    Example for running a movement primitive based environment, which is already registered
    Args:
        env_name: DMP env_id
        seed: seed for deterministic behaviour
        iterations: Number of rollout steps to run
        render: Render the episode

    Returns:

    """
    # Changing the arguments of the black box env is possible by providing them to gym as with all kwargs.
    # E.g. here for way to many basis functions
    env = fancy_gym.make(env_name, seed, basis_generator_kwargs={'num_basis': 1000})
    # env = fancy_gym.make(env_name, seed)
    # mp_dict.update({'black_box_kwargs': {'learn_sub_trajectories': True}})
    # mp_dict.update({'black_box_kwargs': {'do_replanning': lambda pos, vel, t: lambda t: t % 100}})

    returns = 0
    obs = env.reset()

    # This time rendering every trajectory
    if render:
        env.render(mode="human")

    # number of samples/full trajectories (multiple environment steps)
    for i in range(iterations):
        ac = env.action_space.sample()
        obs, reward, done, info = env.step(ac)
        returns += reward

        if done:
            print(i, reward)
            obs = env.reset()

    return obs


def example_fully_custom_mp(seed=1, iterations=1, render=True):
    """
    Example for running a custom movement primitive based environments.
    Our already registered environments follow the same structure.
    Hence, this also allows to adjust hyperparameters of the movement primitives.
    Yet, we recommend the method above if you are just interested in changing those parameters for existing tasks.
    We appreciate PRs for custom environments (especially MP wrappers of existing tasks) 
    for our repo: https://github.com/ALRhub/fancy_gym/
    Args:
        seed: seed
        iterations: Number of rollout steps to run
        render: Render the episode

    Returns:

    """

    base_env_id = "Reacher5d-v0"

    # Replace this wrapper with the custom wrapper for your environment by inheriting from the RawInterfaceWrapper.
    # You can also add other gym.Wrappers in case they are needed.
    wrappers = [fancy_gym.envs.mujoco.reacher.MPWrapper]

    # For a ProMP
    trajectory_generator_kwargs = {'trajectory_generator_type': 'promp',
                                   'weight_scale': 2}
    phase_generator_kwargs = {'phase_generator_type': 'linear'}
    controller_kwargs = {'controller_type': 'velocity'}
    basis_generator_kwargs = {'basis_generator_type': 'zero_rbf',
                              'num_basis': 5,
                              'num_basis_zero_start': 1
                              }

    # # For a DMP
    # trajectory_generator_kwargs = {'trajectory_generator_type': 'dmp',
    #                                'weight_scale': 500}
    # phase_generator_kwargs = {'phase_generator_type': 'exp',
    #                           'alpha_phase': 2.5}
    # controller_kwargs = {'controller_type': 'velocity'}
    # basis_generator_kwargs = {'basis_generator_type': 'rbf',
    #                           'num_basis': 5
    #                           }
    env = fancy_gym.make_bb(env_id=base_env_id, wrappers=wrappers, black_box_kwargs={},
                            traj_gen_kwargs=trajectory_generator_kwargs, controller_kwargs=controller_kwargs,
                            phase_kwargs=phase_generator_kwargs, basis_kwargs=basis_generator_kwargs,
                            seed=seed)

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
    example_mp("HoleReacherDMP-v0", seed=10, iterations=5, render=render)

    # ProMP
    example_mp("HoleReacherProMP-v0", seed=10, iterations=5, render=render)
    example_mp("BoxPushingTemporalSparseProMP-v0", seed=10, iterations=1, render=render)

    # ProDMP
    example_mp("BoxPushingDenseReplanProDMP-v0", seed=10, iterations=4, render=render)

    # Altered basis functions
    obs1 = example_custom_mp("Reacher5dProMP-v0", seed=10, iterations=1, render=render)

    # Custom MP
    example_fully_custom_mp(seed=10, iterations=1, render=render)
