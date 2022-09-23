import numpy as np

import pytest

from dm_control import suite, manipulation

DMC_ENVS = [f'{env}-{task}' for env, task in suite.ALL_TASKS if env != "lqr"]
MANIPULATION_SPECS = [f'manipulation-{task}' for task in manipulation.ALL if task.endswith('_features')]
SEED = 1


@pytest.mark.parametrize('env_id', DMC_ENVS)
def test_dmc_functionality(self, env_id: str):
    """Tests that environments runs without errors using random actions."""
    self.run_env(env_id)


@pytest.mark.parametrize('env_id', DMC_ENVS)
def test_dmc_determinism(self, env_id: str):
    """Tests that identical seeds produce identical trajectories."""
    seed = 0
    self._run_env_determinism(env_id, seed)


@pytest.mark.parametrize('env_id', MANIPULATION_SPECS)
def test_manipulation_functionality(self, env_id: str):
    """Tests that environments runs without errors using random actions."""
    self.run_env(env_id)


@pytest.mark.parametrize('env_id', MANIPULATION_SPECS)
def test_manipulation_determinism(self, env_id: str):
    """Tests that identical seeds produce identical trajectories."""
    seed = 0
    # Iterate over two trajectories, which should have the same state and action sequence
    traj1 = self.run_env(env_id, seed=seed)
    traj2 = self.run_env(env_id, seed=seed)
    for i, time_step in enumerate(zip(*traj1, *traj2)):
        obs1, rwd1, done1, obs2, rwd2, done2 = time_step
        assert np.array_equal(obs1, obs2), f"Observations [{i}] {obs1} and {obs2} do not match."
        assert np.all(rwd1 == rwd2), f"Rewards [{i}] {rwd1} and {rwd2} do not match."
        assert np.all(done1 == done2), f"Dones [{i}] {done1} and {done2} do not match."
