import gym
import numpy as np
from fancy_gym import make


def run_env(env_id, iterations=None, seed=0, render=False):
    """
    Example for running a DMC based env in the step based setting.
    The env_id has to be specified as `dmc:domain_name-task_name` or
    for manipulation tasks as `manipulation-environment_name`

    Args:
        env_id: Either `dmc:domain_name-task_name` or `dmc:manipulation-environment_name`
        iterations: Number of rollout steps to run
        seed: random seeding
        render: Render the episode

    Returns: observations, rewards, dones, actions

    """
    env: gym.Env = make(env_id, seed=seed)
    rewards = []
    observations = []
    actions = []
    dones = []
    obs = env.reset()
    verify_observations(obs, env.observation_space, "reset()")

    iterations = iterations or (env.spec.max_episode_steps or 1)

    # number of samples(multiple environment steps)
    for i in range(iterations):
        observations.append(obs)

        ac = env.action_space.sample()
        actions.append(ac)
        # ac = np.random.uniform(env.action_space.low, env.action_space.high, env.action_space.shape)
        obs, reward, done, info = env.step(ac)

        verify_observations(obs, env.observation_space, "step()")
        verify_reward(reward)
        verify_done(done)

        rewards.append(reward)
        dones.append(done)

        if render:
            env.render("human")

        if done:
            break
    if not hasattr(env, "replanning_schedule"):
        assert done, "Done flag is not True after end of episode."
    observations.append(obs)
    env.close()
    del env
    return np.array(observations), np.array(rewards), np.array(dones), np.array(actions)


def run_env_determinism(env_id: str, seed: int):
    traj1 = run_env(env_id, seed=seed)
    traj2 = run_env(env_id, seed=seed)
    # Iterate over two trajectories, which should have the same state and action sequence
    for i, time_step in enumerate(zip(*traj1, *traj2)):
        obs1, rwd1, done1, ac1, obs2, rwd2, done2, ac2 = time_step
        assert np.array_equal(obs1, obs2), f"Observations [{i}] {obs1} and {obs2} do not match."
        assert np.array_equal(ac1, ac2), f"Actions [{i}] {ac1} and {ac2} do not match."
        assert np.array_equal(rwd1, rwd2), f"Rewards [{i}] {rwd1} and {rwd2} do not match."
        assert np.array_equal(done1, done2), f"Dones [{i}] {done1} and {done2} do not match."


def verify_observations(obs, observation_space: gym.Space, obs_type="reset()"):
    assert observation_space.contains(obs), \
        f"Observation {obs} received from {obs_type} not contained in observation space {observation_space}."


def verify_reward(reward):
    assert isinstance(reward, (float, int)), f"Returned type {type(reward)} as reward, expected float or int."


def verify_done(done):
    assert isinstance(done, bool), f"Returned {done} as done flag, expected bool."
