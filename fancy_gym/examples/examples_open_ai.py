import gymnasium as gym
import fancy_gym


def example_mp(env_name, seed=1, render=True):
    """
    Example for running a movement primitive based version of a OpenAI-gym environment, which is already registered.
    For more information on movement primitive specific stuff, look at the traj_gen examples.
    Args:
        env_name: ProMP env_id
        seed: seed
        render: boolean
    Returns:

    """
    env = gym.make(env_name, render_mode='human' if render else None)

    returns = 0
    obs = env.reset(seed=seed)
    # number of samples/full trajectories (multiple environment steps)
    for i in range(10):
        if render and i % 2 == 0:
            env.render()
        ac = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(ac)
        returns += reward

        if terminated or truncated:
            print(returns)
            obs = env.reset()

def main(render=True):
    example_mp("gym_ProMP/Reacher-v2", render=render)

if __name__ == '__main__':
    main()