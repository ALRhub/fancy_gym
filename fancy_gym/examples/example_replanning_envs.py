import gymnasium as gym
import fancy_gym


def example_run_replanning_env(env_name="fancy_ProDMP/BoxPushingDenseReplan-v0", seed=1, iterations=1, render=False):
    env = gym.make(env_name)
    env.reset(seed=seed)
    for i in range(iterations):
        done = False
        while done is False:
            ac = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(ac)
            if render:
                env.render(mode="human")
            if terminated or truncated:
                env.reset()
    env.close()
    del env


def example_custom_replanning_envs(seed=0, iteration=100, render=True):
    # id for a step-based environment
    base_env_id = "BoxPushingDense-v0"

    wrappers = [fancy_gym.envs.mujoco.box_pushing.mp_wrapper.MPWrapper]

    trajectory_generator_kwargs = {'trajectory_generator_type': 'prodmp',
                                   'weights_scale': 1}
    phase_generator_kwargs = {'phase_generator_type': 'exp'}
    controller_kwargs = {'controller_type': 'velocity'}
    basis_generator_kwargs = {'basis_generator_type': 'prodmp',
                              'num_basis': 5}

    # max_planning_times: the maximum number of plans can be generated
    # replanning_schedule: the trigger for replanning
    # condition_on_desired: use desired state as the boundary condition for the next plan
    black_box_kwargs = {'max_planning_times': 4,
                        'replanning_schedule': lambda pos, vel, obs, action, t: t % 25 == 0,
                        'condition_on_desired': True}

    env = fancy_gym.make_bb(env_id=base_env_id, wrappers=wrappers, black_box_kwargs=black_box_kwargs,
                            traj_gen_kwargs=trajectory_generator_kwargs, controller_kwargs=controller_kwargs,
                            phase_kwargs=phase_generator_kwargs, basis_kwargs=basis_generator_kwargs,
                            seed=seed)
    if render:
        env.render(mode="human")

    obs = env.reset()

    for i in range(iteration):
        ac = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(ac)
        if terminated or truncated:
            env.reset()

    env.close()
    del env


if __name__ == "__main__":
    # run a registered replanning environment
    example_run_replanning_env(env_name="fancy_ProDMP/BoxPushingDenseReplan-v0", seed=1, iterations=1, render=False)

    # run a custom replanning environment
    example_custom_replanning_envs(seed=0, iteration=8, render=True)
