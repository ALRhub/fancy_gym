import gymnasium as gym
import fancy_gym


def example_run_replanning_env(env_name="fancy_ProDMP/BoxPushingDenseReplan-v0", seed=1, iterations=1, render=False):
    env = gym.make(env_name, render_mode='human' if render else None)
    env.reset(seed=seed)
    for i in range(iterations):
        while True:
            ac = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(ac)
            if render:
                env.render()
            if terminated or truncated:
                env.reset()
                break
    env.close()
    del env


def example_custom_replanning_envs(seed=0, iteration=100, render=True):
    # id for a step-based environment
    base_env_id = "fancy/BoxPushingDense-v0"

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

    base_env = gym.make(base_env_id, render_mode='human' if render else None)
    env = fancy_gym.make_bb(env=base_env, wrappers=wrappers, black_box_kwargs=black_box_kwargs,
                            traj_gen_kwargs=trajectory_generator_kwargs, controller_kwargs=controller_kwargs,
                            phase_kwargs=phase_generator_kwargs, basis_kwargs=basis_generator_kwargs,
                            seed=seed)
    if render:
        env.render()

    obs = env.reset()

    for i in range(iteration):
        ac = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(ac)
        if terminated or truncated:
            env.reset()

    env.close()
    del env

def main(render=False):
    # run a registered replanning environment
    example_run_replanning_env(env_name="fancy_ProDMP/BoxPushingDenseReplan-v0", seed=1, iterations=1, render=render)

    # run a custom replanning environment
    example_custom_replanning_envs(seed=0, iteration=8, render=render)

if __name__ == "__main__":
    main()