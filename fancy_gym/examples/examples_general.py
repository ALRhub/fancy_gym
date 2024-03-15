from collections import defaultdict

import gymnasium as gym
import numpy as np

import fancy_gym


def example_general(env_id="Pendulum-v1", seed=1, iterations=1000, render=True):
    """
    Example for running any env in the step based setting.
    This also includes DMC environments when leveraging our custom make_env function.

    Args:
        env_id: OpenAI/Custom gym task id or either `domain_name-task_name` or `manipulation-environment_name` for DMC tasks
        seed: seed for deterministic behaviour
        iterations: Number of rollout steps to run
        render: Render the episode

    Returns:

    """

    env = gym.make(env_id, render_mode='human' if render else None)
    rewards = 0
    obs = env.reset(seed=seed)
    print("Observation shape: ", env.observation_space.shape)
    print("Action shape: ", env.action_space.shape)

    # number of environment steps
    for i in range(iterations):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        rewards += reward

        if render:
            env.render()

        if terminated or truncated:
            print(rewards)
            rewards = 0
            obs = env.reset()


def example_async(env_id="fancy/HoleReacher-v0", n_cpu=4, seed=int('533D', 16), n_samples=800):
    """
    Example for running any env in a vectorized multiprocessing setting to generate more samples faster.
    This also includes DMC and DMP environments when leveraging our custom make_env function.
    Be aware, increasing the number of environments reduces the total length of the individual episodes.

    Args:
        env_id: OpenAI/Custom gym task id or either `domain_name-task_name` or `manipulation-environment_name` for DMC tasks
        seed: seed for deterministic behaviour
        n_cpu: Number of cpus cores to use in parallel
        n_samples: number of samples generated in total by all environments.

    Returns: Tuple of (obs, reward, done, info) with type np.ndarray

    """
    env = gym.vector.AsyncVectorEnv([fancy_gym.make_rank(env_id, seed, i) for i in range(n_cpu)])
    # OR
    # envs = gym.vector.AsyncVectorEnv([make_env(env_id, seed + i) for i in range(n_cpu)])

    # for plotting
    rewards = np.zeros(n_cpu)
    buffer = defaultdict(list)

    obs = env.reset()

    # this would generate more samples than requested if n_samples % num_envs != 0
    repeat = int(np.ceil(n_samples / env.num_envs))
    for i in range(repeat):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        buffer['obs'].append(obs)
        buffer['reward'].append(reward)
        buffer['terminated'].append(terminated)
        buffer['truncated'].append(truncated)
        buffer['info'].append(info)
        rewards += reward

        done = terminated or truncated
        if np.any(done):
            print(f"Reward at iteration {i}: {rewards[done]}")
            rewards[done] = 0

    # do not return values above threshold
    return *map(lambda v: np.stack(v)[:n_samples], buffer.values()),

def main(render = False):
    # Basic gym task
    example_general("Pendulum-v1", seed=10, iterations=200, render=render)

    # Mujoco task from framework
    example_general("fancy/Reacher5d-v0", seed=10, iterations=200, render=render)

    # # OpenAI Mujoco task
    example_general("HalfCheetah-v2", seed=10, render=render)

    # Vectorized multiprocessing environments
    # example_async(env_id="HoleReacher-v0", n_cpu=2, seed=int('533D', 16), n_samples=2 * 200)

if __name__ == '__main__':
    main()