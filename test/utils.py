from typing import List, Type

import gymnasium as gym
import numpy as np
from gymnasium import make


def run_env(env_id: str, iterations: int = None, seed: int = 0, wrappers: List[Type[gym.Wrapper]] = [],
            render: bool = False):
    """
    Example for running a DMC based env in the step based setting.
    The env_id has to be specified as `dmc:domain_name-task_name` or
    for manipulation tasks as `manipulation-environment_name`

    Args:
        env_id: Either `dmc:domain_name-task_name` or `dmc:manipulation-environment_name`
        iterations: Number of rollout steps to run
        seed: random seeding
        wrappers: List of Wrappers to apply to the environment
        render: Render the episode

    Returns: observations, rewards, terminations, truncations, actions

    """
    env: gym.Env = make(env_id)
    for w in wrappers:
        env = w(env)
    rewards = []
    observations = []
    actions = []
    terminations = []
    truncations = []
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    verify_observations(obs, env.observation_space, "reset()")

    iterations = iterations or (env.spec.max_episode_steps or 1)

    # number of samples (multiple environment steps)
    for i in range(iterations):
        observations.append(obs)

        ac = env.action_space.sample()
        actions.append(ac)
        # ac = np.random.uniform(env.action_space.low, env.action_space.high, env.action_space.shape)
        obs, reward, terminated, truncated, info = env.step(ac)

        verify_observations(obs, env.observation_space, "step()")
        verify_reward(reward)
        verify_done(terminated)
        verify_done(truncated)

        rewards.append(reward)
        terminations.append(terminated)
        truncations.append(truncated)

        if render:
            env.render("human")

        if terminated or truncated:
            break
    if not hasattr(env, "replanning_schedule"):
        assert terminated or truncated, f"Termination or truncation flag is not True after {i + 1} iterations."

    observations.append(obs)
    env.close()
    del env
    return np.array(observations), np.array(rewards), np.array(terminations), np.array(truncations), np.array(actions)


def run_env_determinism(env_id: str, seed: int, iterations: int = None, wrappers: List[Type[gym.Wrapper]] = []):
    traj1 = run_env(env_id, iterations=iterations,
                    seed=seed, wrappers=wrappers)
    traj2 = run_env(env_id, iterations=iterations,
                    seed=seed, wrappers=wrappers)
    # Iterate over two trajectories, which should have the same state and action sequence
    for i, time_step in enumerate(zip(*traj1, *traj2)):
        obs1, rwd1, term1, trunc1, ac1, obs2, rwd2, term2, trunc2, ac2 = time_step
        assert np.allclose(
            obs1, obs2), f"Observations [{i}] {obs1} ({obs1.shape}) and {obs2} ({obs2.shape}) do not match: Biggest difference is {np.abs(obs1-obs2).max()} at index {np.abs(obs1-obs2).argmax()}."
        assert np.array_equal(
            ac1, ac2), f"Actions [{i}] {ac1} and {ac2} do not match."
        assert np.array_equal(
            rwd1, rwd2), f"Rewards [{i}] {rwd1} and {rwd2} do not match."
        assert np.array_equal(
            term1, term2), f"Terminateds [{i}] {term1} and {term2} do not match."
        assert np.array_equal(
            term1, term2), f"Truncateds [{i}] {trunc1} and {trunc2} do not match."


def verify_observations(obs, observation_space: gym.Space, obs_type="reset()"):
    assert observation_space.contains(obs), \
        f"Observation {obs} ({obs.shape}) received from {obs_type} not contained in observation space {observation_space}."


def verify_reward(reward):
    assert isinstance(
        reward, (float, int)), f"Returned type {type(reward)} as reward, expected float or int."


def verify_done(done):
    assert isinstance(
        done, bool), f"Returned {done} as done flag, expected bool."
