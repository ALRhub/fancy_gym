from alr_envs import HoleReacherMPWrapper
from alr_envs.utils.make_env_helpers import make_dmp_env, make_env


def example_mp(env_name="alr_envs:HoleReacherDMP-v1", seed=1):
    """
    Example for running a motion primitive based environment, which is already registered
    Args:
        env_name: DMP env_id
        seed: seed

    Returns:

    """
    # While in this case gym.make() is possible to use as well, we recommend our custom make env function.
    # First, it already takes care of seeding and second enables the use of DMC tasks within the gym interface.
    env = make_env(env_name, seed)
    rewards = 0
    # env.render(mode=None)
    obs = env.reset()

    # number of samples/full trajectories (multiple environment steps)
    for i in range(10):
        ac = env.action_space.sample()
        obs, reward, done, info = env.step(ac)
        rewards += reward

        if i % 1 == 0:
            # render full DMP trajectory
            # render can only be called once in the beginning as well. That would render every trajectory
            # Calling it after every trajectory allows to modify the mode. mode=None, disables rendering.
            env.render(mode="human")

        if done:
            print(rewards)
            rewards = 0
            obs = env.reset()


def example_custom_mp(seed=1):
    """
    Example for running a custom motion primitive based environments.
    Our already registered environments follow the same structure, but do not directly allow for modifications.
    Hence, this also allows to adjust hyperparameters of the motion primitives more easily.
    We appreciate PRs for custom environments (especially MP wrappers of existing tasks) 
    for our repo: https://github.com/ALRhub/alr_envs/
    Args:
        seed: seed

    Returns:

    """

    base_env = "alr_envs:HoleReacher-v1"
    # Replace this wrapper with the custom wrapper for your environment by inheriting from the MPEnvWrapper.
    # You can also add other gym.Wrappers in case they are needed.
    wrappers = [HoleReacherMPWrapper]
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
    env = make_dmp_env(base_env, wrappers=wrappers, seed=seed, **mp_kwargs)
    # OR for a deterministic ProMP:
    # env = make_detpmp_env(base_env, wrappers=wrappers, seed=seed)

    rewards = 0
    # env.render(mode=None)
    obs = env.reset()

    # number of samples/full trajectories (multiple environment steps)
    for i in range(10):
        ac = env.action_space.sample()
        obs, reward, done, info = env.step(ac)
        rewards += reward

        if i % 1 == 0:
            # render full DMP trajectory
            # render can only be called once in the beginning as well. That would render every trajectory
            # Calling it after every trajectory allows to modify the mode. mode=None, disables rendering.
            env.render(mode="human")

        if done:
            print(rewards)
            rewards = 0
            obs = env.reset()


if __name__ == '__main__':
    # DMP
    example_mp("alr_envs:HoleReacherDMP-v1")

    # DetProMP
    example_mp("alr_envs:HoleReacherDetPMP-v1")

    # Custom DMP
    example_custom_mp()
