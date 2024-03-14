import gymnasium as gym
import fancy_gym


def example_dmc(env_id="dm_control/fish-swim", seed=1, iterations=1000, render=True):
    """
    Example for running a DMC based env in the step based setting.
    The env_id has to be specified as `domain_name:task_name` or
    for manipulation tasks as `domain_name:manipulation-environment_name`

    Args:
        env_id: Either `domain_name-task_name` or `manipulation-environment_name`
        seed: seed for deterministic behaviour
        iterations: Number of rollout steps to run
        render: Render the episode

    Returns:

    """
    env = gym.make(env_id, render_mode='human' if render else None)
    rewards = 0
    obs = env.reset(seed=seed)
    print("observation shape:", env.observation_space.shape)
    print("action shape:", env.action_space.shape)

    for i in range(iterations):
        ac = env.action_space.sample()
        if render:
            env.render()
        obs, reward, terminated, truncated, info = env.step(ac)
        rewards += reward

        if terminated or truncated:
            print(env_id, rewards)
            rewards = 0
            obs = env.reset()

    env.close()
    del env


def example_custom_dmc_and_mp(seed=1, iterations=1, render=True):
    """
    Example for running a custom movement primitive based environments.
    Our already registered environments follow the same structure.
    Hence, this also allows to adjust hyperparameters of the movement primitives.
    Yet, we recommend the method above if you are just interested in chaining those parameters for existing tasks.
    We appreciate PRs for custom environments (especially MP wrappers of existing tasks)
    for our repo: https://github.com/ALRhub/fancy_gym/
    Args:
        seed: seed for deterministic behaviour
        iterations: Number of rollout steps to run
        render: Render the episode

    Returns:

    """

    # Base DMC name, according to structure of above example
    base_env_id = "dm_control/ball_in_cup-catch"

    # Replace this wrapper with the custom wrapper for your environment by inheriting from the RawInterfaceWrapper.
    # You can also add other gym.Wrappers in case they are needed.
    wrappers = [fancy_gym.dmc.suite.ball_in_cup.MPWrapper]
    # # For a ProMP
    trajectory_generator_kwargs = {'trajectory_generator_type': 'promp'}
    phase_generator_kwargs = {'phase_generator_type': 'linear'}
    controller_kwargs = {'controller_type': 'motor',
                         "p_gains": 1.0,
                         "d_gains": 0.1, }
    basis_generator_kwargs = {'basis_generator_type': 'zero_rbf',
                              'num_basis': 5,
                              'num_basis_zero_start': 1
                              }

    # For a DMP
    # trajectory_generator_kwargs = {'trajectory_generator_type': 'dmp'}
    # phase_generator_kwargs = {'phase_generator_type': 'exp',
    #                           'alpha_phase': 2}
    # controller_kwargs = {'controller_type': 'motor',
    #                      "p_gains": 1.0,
    #                      "d_gains": 0.1,
    #                      }
    # basis_generator_kwargs = {'basis_generator_type': 'rbf',
    #                           'num_basis': 5
    #                           }
    base_env = gym.make(base_env_id, render_mode='human' if render else None)
    env = fancy_gym.make_bb(env=base_env, wrappers=wrappers, black_box_kwargs={},
                            traj_gen_kwargs=trajectory_generator_kwargs, controller_kwargs=controller_kwargs,
                            phase_kwargs=phase_generator_kwargs, basis_kwargs=basis_generator_kwargs,
                            seed=seed)

    # This renders the full MP trajectory
    # It is only required to call render() once in the beginning, which renders every consecutive trajectory.
    # Resetting to no rendering, can be achieved by render(mode=None).
    # It is also possible to change them mode multiple times when
    # e.g. only every nth trajectory should be displayed.
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
            print(base_env_id, rewards)
            rewards = 0
            obs = env.reset()

    env.close()
    del env

def main(render = False):
    # # Standard DMC Suite tasks
    example_dmc("dm_control/fish-swim", seed=10, iterations=1000, render=render)
    #
    # # Manipulation tasks
    # # Disclaimer: The vision versions are currently not integrated and yield an error
    example_dmc("dm_control/reach_site_features", seed=10, iterations=250, render=render)
    #
    # # Gym + DMC hybrid task provided in the MP framework
    example_dmc("dm_control_ProMP/ball_in_cup-catch-v0", seed=10, iterations=1, render=render)

    # Custom DMC task # Different seed, because the episode is longer for this example and the name+seed combo is
    # already registered above
    example_custom_dmc_and_mp(seed=11, iterations=1, render=render)

    # # Standard DMC Suite tasks
    example_dmc("dm_control/fish-swim", seed=10, iterations=1000, render=render)
    #
    # # Manipulation tasks
    # # Disclaimer: The vision versions are currently not integrated and yield an error
    example_dmc("dm_control/reach_site_features", seed=10, iterations=250, render=render)
    #
    # # Gym + DMC hybrid task provided in the MP framework
    example_dmc("dm_control_ProMP/ball_in_cup-catch-v0", seed=10, iterations=1, render=render)

    # Custom DMC task # Different seed, because the episode is longer for this example and the name+seed combo is
    # already registered above
    example_custom_dmc_and_mp(seed=11, iterations=1, render=render)

if __name__ == '__main__':
    main()