import gym
import numpy as np
from fancy_gym import make


def run_env(env_id, iterations=None, seed=SEED, render=False):
    """
    Example for running a DMC based env in the step based setting.
    The env_id has to be specified as `domain_name-task_name` or
    for manipulation tasks as `manipulation-environment_name`

    Args:
        env_id: Either `domain_name-task_name` or `manipulation-environment_name`
        iterations: Number of rollout steps to run
        seed= random seeding
        render: Render the episode

    Returns:

    """
    env: gym.Env = make(env_id, seed=seed)
    rewards = []
    observations = []
    dones = []
    obs = env.reset()
    _verify_observations(obs, env.observation_space, "reset()")

    length = env.spec.max_episode_steps
    if iterations is None:
        if length is None:
            iterations = 1
        else:
            iterations = length

    # number of samples(multiple environment steps)
    for i in range(iterations):
        observations.append(obs)

        ac = env.action_space.sample()
        # ac = np.random.uniform(env.action_space.low, env.action_space.high, env.action_space.shape)
        obs, reward, done, info = env.step(ac)

        _verify_observations(obs, env.observation_space, "step()")
        _verify_reward(reward)
        _verify_done(done)

        rewards.append(reward)
        dones.append(done)

        if render:
            env.render("human")

        if done:
            obs = env.reset()

    assert done, "Done flag is not True after max episode length."
    observations.append(obs)
    env.close()
    del env
    return np.array(observations), np.array(rewards), np.array(dones)


def _run_env_determinism(self, env_id: str, seed: int):
    traj1 = self.run_env(env_id, seed=seed)
    traj2 = self.run_env(env_id, seed=seed)
    # Iterate over two trajectories, which should have the same state and action sequence
    for i, time_step in enumerate(zip(*traj1, *traj2)):
        obs1, rwd1, done1, obs2, rwd2, done2 = time_step
        self.assertTrue(np.array_equal(obs1, obs2), f"Observations [{i}] {obs1} and {obs2} do not match.")
        self.assertEqual(rwd1, rwd2, f"Rewards [{i}] {rwd1} and {rwd2} do not match.")
        self.assertEqual(done1, done2, f"Dones [{i}] {done1} and {done2} do not match.")


def _verify_observations(obs, observation_space, obs_type="reset()"):
    assert observation_space.contains(obs), \
        f"Observation {obs} received from {obs_type} not contained in observation space {observation_space}."


def _verify_reward(reward):
    assert isinstance(reward, float), f"Returned {reward} as reward, expected float."


def _verify_done(done):
    assert isinstance(done, bool), f"Returned {done} as done flag, expected bool."
