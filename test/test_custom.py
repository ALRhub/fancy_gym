import unittest

import gym
import numpy as np

import alr_envs  # noqa
from alr_envs.utils.make_env_helpers import make

CUSTOM_IDS = [spec.id for spec in gym.envs.registry.all() if
              "alr_envs" in spec.entry_point and not 'make_bb_env_helper' in spec.entry_point]
SEED = 1


class TestCustomEnvironments(unittest.TestCase):

    def _run_env(self, env_id, iterations=None, seed=SEED, render=False):
        """
        Example for running a DMC based env in the step based setting.
        The env_id has to be specified as `domain_name-task_name` or
        for manipulation tasks as `manipulation-environment_name`

        Args:
            env_id: Either `domain_name-task_name` or `manipulation-environment_name`
            iterations: Number of rollout steps to run
            seed: random seeding
            render: Render the episode

        Returns: observations, rewards, dones, actions

        """
        env: gym.Env = make(env_id, seed=seed)
        rewards = []
        actions = []
        observations = []
        dones = []
        obs = env.reset()
        self._verify_observations(obs, env.observation_space, "reset()")

        iterations = iterations or (env.spec.max_episode_steps or 1)

        # number of samples(multiple environment steps)
        for i in range(iterations):
            observations.append(obs)

            ac = env.action_space.sample()
            actions.append(ac)
            obs, reward, done, info = env.step(ac)

            self._verify_observations(obs, env.observation_space, "step()")
            self._verify_reward(reward)
            self._verify_done(done)

            rewards.append(reward)
            dones.append(done)

            if render:
                env.render("human")

            if done:
                break

        assert done, "Done flag is not True after end of episode."
        observations.append(obs)
        env.close()
        del env
        return np.array(observations), np.array(rewards), np.array(dones), np.array(actions)

    def _run_env_determinism(self, ids):
        seed = 0
        for env_id in ids:
            with self.subTest(msg=env_id):
                traj1 = self._run_env(env_id, seed=seed)
                traj2 = self._run_env(env_id, seed=seed)
                for i, time_step in enumerate(zip(*traj1, *traj2)):
                    obs1, rwd1, done1, ac1, obs2, rwd2, done2, ac2 = time_step
                    self.assertTrue(np.array_equal(ac1, ac2), f"Actions [{i}] delta {ac1 - ac2} is not zero.")
                    self.assertTrue(np.array_equal(obs1, obs2), f"Observations [{i}] delta {obs1 - obs2} is not zero.")
                    self.assertEqual(rwd1, rwd2, f"Rewards [{i}] {rwd1} and {rwd2} do not match.")
                    self.assertEqual(done1, done2, f"Dones [{i}] {done1} and {done2} do not match.")

    def _verify_observations(self, obs, observation_space, obs_type="reset()"):
        self.assertTrue(observation_space.contains(obs),
                        f"Observation {obs} received from {obs_type} "
                        f"not contained in observation space {observation_space}.")

    def _verify_reward(self, reward):
        self.assertIsInstance(reward, (float, int), f"Returned type {type(reward)} as reward, expected float or int.")

    def _verify_done(self, done):
        self.assertIsInstance(done, bool, f"Returned {done} as done flag, expected bool.")

    def test_step_functionality(self):
        """Tests that step environments run without errors using random actions."""
        for env_id in CUSTOM_IDS:
            with self.subTest(msg=env_id):
                self._run_env(env_id)

    def test_step_determinism(self):
        """Tests that for step environments identical seeds produce identical trajectories."""
        self._run_env_determinism(CUSTOM_IDS)

    def test_bb_functionality(self):
        """Tests that black box environments run without errors using random actions."""
        for traj_gen, env_ids in alr_envs.ALL_ALR_MOVEMENT_PRIMITIVE_ENVIRONMENTS.items():
            with self.subTest(msg=traj_gen):
                for id in env_ids:
                    with self.subTest(msg=id):
                        self._run_env(id)

    def test_bb_determinism(self):
        """Tests that for black box environment identical seeds produce identical trajectories."""
        for traj_gen, env_ids in alr_envs.ALL_ALR_MOVEMENT_PRIMITIVE_ENVIRONMENTS.items():
            with self.subTest(msg=traj_gen):
                self._run_env_determinism(env_ids)


if __name__ == '__main__':
    unittest.main()
