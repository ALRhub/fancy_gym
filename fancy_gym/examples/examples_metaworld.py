import gymnasium as gym
import fancy_gym


def example_meta(env_id="metaworld/button-press-v2", seed=1, iterations=1000, render=True):
    """
    Example for running a MetaWorld based env in the step based setting.
    The env_id has to be specified as `task_name-v2`. V1 versions are not supported and we always
    return the observable goal version.
    All tasks can be found here: https://arxiv.org/pdf/1910.10897.pdf or https://meta-world.github.io/

    Args:
        env_id: `task_name-v2`
        seed: seed for deterministic behaviour (TODO: currently not working due to an issue in MetaWorld code)
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
            obs = env.reset(seed=seed+i+1)

    env.close()
    del env


def example_custom_meta_and_mp(seed=1, iterations=1, render=True):
    """
    Example for running a custom movement primitive based environments.
    Our already registered environments follow the same structure.
    Hence, this also allows to adjust hyperparameters of the movement primitives.
    Yet, we recommend the method above if you are just interested in chaining those parameters for existing tasks.
    We appreciate PRs for custom environments (especially MP wrappers of existing tasks)
    for our repo: https://github.com/ALRhub/fancy_gym/
    Args:
        seed: seed for deterministic behaviour (TODO: currently not working due to an issue in MetaWorld code)
        iterations: Number of rollout steps to run
        render: Render the episode (TODO: currently not working due to an issue in MetaWorld code)

    Returns:

    """

    # Base MetaWorld name, according to structure of above example
    base_env_id = "metaworld/button-press-v2"

    # Replace this wrapper with the custom wrapper for your environment by inheriting from the RawInterfaceWrapper.
    # You can also add other gym.Wrappers in case they are needed.
    wrappers = [fancy_gym.meta.goal_object_change_mp_wrapper.MPWrapper]
    # # For a ProMP
    # trajectory_generator_kwargs = {'trajectory_generator_type': 'promp'}
    # phase_generator_kwargs = {'phase_generator_type': 'linear'}
    # controller_kwargs = {'controller_type': 'metaworld'}
    # basis_generator_kwargs = {'basis_generator_type': 'zero_rbf',
    #                           'num_basis': 5,
    #                           'num_basis_zero_start': 1
    #                           }

    # For a DMP
    trajectory_generator_kwargs = {'trajectory_generator_type': 'dmp'}
    phase_generator_kwargs = {'phase_generator_type': 'exp',
                              'alpha_phase': 2}
    controller_kwargs = {'controller_type': 'metaworld'}
    basis_generator_kwargs = {'basis_generator_type': 'rbf',
                              'num_basis': 5
                              }
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
    obs = env.reset(seed=seed)

    # number of samples/full trajectories (multiple environment steps)
    for i in range(iterations):
        ac = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(ac)
        rewards += reward

        if terminated or truncated:
            print(base_env_id, rewards)
            rewards = 0
            obs = env.reset(seed=seed+i+1)

    env.close()
    del env

def main(render = False):
    # For rendering it might be necessary to specify your OpenGL installation
    # export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so

    # # Standard Meta world tasks
    example_meta("metaworld/button-press-v2", seed=10, iterations=500, render=render)

    # # MP + MetaWorld hybrid task provided in the our framework
    example_meta("metaworld_ProMP/button-press-v2", seed=10, iterations=1, render=render)
    #
    # # Custom MetaWorld task
    example_custom_meta_and_mp(seed=10, iterations=1, render=render)

if __name__ == '__main__':
    main()