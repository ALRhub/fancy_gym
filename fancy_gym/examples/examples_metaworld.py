import gymnasium as gym
import fancy_gym


def example_meta(env_id="fish-swim", seed=1, iterations=1000, render=True):
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
    env = gym.make(env_id)
    rewards = 0
    obs = env.reset(seed=seed)
    print("observation shape:", env.observation_space.shape)
    print("action shape:", env.action_space.shape)

    for i in range(iterations):
        ac = env.action_space.sample()
        if render:
            # THIS NEEDS TO BE SET TO FALSE FOR NOW, BECAUSE THE INTERFACE FOR RENDERING IS DIFFERENT TO BASIC GYM
            # TODO: Remove this, when Metaworld fixes its interface.
            env.render(False)
        obs, reward, terminated, truncated, info = env.step(ac)
        rewards += reward
        if terminated or truncated:
            print(env_id, rewards)
            rewards = 0
            obs = env.reset()

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
    env = fancy_gym.make_bb(env_id=base_env_id, wrappers=wrappers, black_box_kwargs={},
                            traj_gen_kwargs=trajectory_generator_kwargs, controller_kwargs=controller_kwargs,
                            phase_kwargs=phase_generator_kwargs, basis_kwargs=basis_generator_kwargs,
                            seed=seed)

    # This renders the full MP trajectory
    # It is only required to call render() once in the beginning, which renders every consecutive trajectory.
    # Resetting to no rendering, can be achieved by render(mode=None).
    # It is also possible to change them mode multiple times when
    # e.g. only every nth trajectory should be displayed.
    if render:
        raise ValueError("Metaworld render interface bug does not allow to render() fixes its interface. "
                         "A temporary workaround is to alter their code in MujocoEnv render() from "
                         "`if not offscreen` to `if not offscreen or offscreen == 'human'`.")
        # TODO: Remove this, when Metaworld fixes its interface.
        # env.render(mode="human")

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


if __name__ == '__main__':
    # Disclaimer: MetaWorld environments require the seed to be specified in the beginning.
    # Adjusting it afterwards with env.seed() is not recommended as it may not affect the underlying behavior.

    # For rendering it might be necessary to specify your OpenGL installation
    # export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
    render = False

    # # Standard Meta world tasks
    example_meta("metaworld/button-press-v2", seed=10, iterations=500, render=render)

    # # MP + MetaWorld hybrid task provided in the our framework
    example_meta("metaworld_ProMP/ButtonPress-v2", seed=10, iterations=1, render=render)
    #
    # # Custom MetaWorld task
    example_custom_meta_and_mp(seed=10, iterations=1, render=render)
