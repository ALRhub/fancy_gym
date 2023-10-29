import gymnasium as gym
import fancy_gym


def example_mp(env_name="fancy_ProMP/HoleReacher-v0", seed=1, iterations=1, render=True):
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
    env = gym.make(env_name, render_mode='human' if render else None)

    returns = 0
    # env.render(mode=None)
    obs = env.reset(seed=seed)

    # number of samples/full trajectories (multiple environment steps)
    for i in range(iterations):

        if render and i % 1 == 0:
            env.render()

        # Now the action space is not the raw action but the parametrization of the trajectory generator,
        # such as a ProMP
        ac = env.action_space.sample()
        # This executes a full trajectory and gives back the context (obs) of the last step in the trajectory, or the
        # full observation space of the last step, if replanning/sub-trajectory learning is used. The 'reward' is equal
        # to the return of a trajectory. Default is the sum over the step-wise rewards.
        obs, reward, terminated, truncated, info = env.step(ac)
        # Aggregated returns
        returns += reward

        if terminated or truncated:
            print(reward)
            obs = env.reset()


def example_custom_mp(env_name="fancy_ProMP/Reacher5d-v0", seed=1, iterations=1, render=True):
    """
    Example for running a movement primitive based environment, which is already registered
    Args:
        env_name: DMP env_id
        seed: seed for deterministic behaviour
        iterations: Number of rollout steps to run
        render: Render the episode

    Returns:

    """
    # Changing the arguments of the black box env is possible by providing them to gym through mp_config_override.
    # E.g. here for way to many basis functions
    env = gym.make(env_name, seed, mp_config_override={'basis_generator_kwargs': {'num_basis': 1000}}, render_mode='human' if render else None)

    returns = 0
    obs = env.reset()

    # This time rendering every trajectory
    if render:
        env.render()

    # number of samples/full trajectories (multiple environment steps)
    for i in range(iterations):
        ac = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(ac)
        returns += reward

        if terminated or truncated:
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

    base_env_id = "fancy/Reacher5d-v0"

    # Replace this wrapper with the custom wrapper for your environment by inheriting from the RawInterfaceWrapper.
    # You can also add other gym.Wrappers in case they are needed.
    wrappers = [fancy_gym.envs.mujoco.reacher.MPWrapper]

    # For a ProMP
    trajectory_generator_kwargs = {'trajectory_generator_type': 'promp',
                                   'weights_scale': 2}
    phase_generator_kwargs = {'phase_generator_type': 'linear'}
    controller_kwargs = {'controller_type': 'velocity'}
    basis_generator_kwargs = {'basis_generator_type': 'zero_rbf',
                              'num_basis': 5,
                              'num_basis_zero_start': 1
                              }

    # # For a DMP
    # trajectory_generator_kwargs = {'trajectory_generator_type': 'dmp',
    #                                'weights_scale': 500}
    # phase_generator_kwargs = {'phase_generator_type': 'exp',
    #                           'alpha_phase': 2.5}
    # controller_kwargs = {'controller_type': 'velocity'}
    # basis_generator_kwargs = {'basis_generator_type': 'rbf',
    #                           'num_basis': 5
    #                           }
    raw_env = gym.make(base_env_id, render_mode='human' if render else None)
    env = fancy_gym.make_bb(env=raw_env, wrappers=wrappers, black_box_kwargs={},
                            traj_gen_kwargs=trajectory_generator_kwargs, controller_kwargs=controller_kwargs,
                            phase_kwargs=phase_generator_kwargs, basis_kwargs=basis_generator_kwargs,
                            seed=seed)

    if render:
        env.render()

    rewards = 0
    obs = env.reset()

    # number of samples/full trajectories (multiple environment steps)
    for i in range(iterations):
        ac = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(ac)
        rewards += reward

        if terminated or truncated:
            print(rewards)
            rewards = 0
            obs = env.reset()


def main():
    render = False
    # DMP
    example_mp("fancy_DMP/HoleReacher-v0", seed=10, iterations=5, render=render)

    # ProMP
    example_mp("fancy_ProMP/HoleReacher-v0", seed=10, iterations=5, render=render)
    example_mp("fancy_ProMP/BoxPushingTemporalSparse-v0", seed=10, iterations=1, render=render)
    example_mp("fancy_ProMP/TableTennis4D-v0", seed=10, iterations=20, render=render)

    # ProDMP with Replanning
    example_mp("fancy_ProDMP/BoxPushingDenseReplan-v0", seed=10, iterations=4, render=render)
    example_mp("fancy_ProDMP/TableTennis4DReplan-v0", seed=10, iterations=20, render=render)
    example_mp("fancy_ProDMP/TableTennisWindReplan-v0", seed=10, iterations=20, render=render)

    # Altered basis functions
    obs1 = example_custom_mp("fancy_ProMP/Reacher5d-v0", seed=10, iterations=1, render=render)

    # Custom MP
    example_fully_custom_mp(seed=10, iterations=1, render=render)

if __name__=='__main__':
    main()
