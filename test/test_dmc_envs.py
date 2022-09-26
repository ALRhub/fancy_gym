from typing import Tuple

import fancy_gym
import pytest
from dm_control import suite, manipulation

from test.utils import run_env_determinism, run_env

SUITE_IDS = [f'dmc:{env}-{task}' for env, task in suite.ALL_TASKS if env != "lqr"]
MANIPULATION_IDS = [f'dmc:manipulation-{task}' for task in manipulation.ALL if task.endswith('_features')]
SEED = 1


@pytest.mark.parametrize('env_id', SUITE_IDS)
def test_step_suite_functionality(env_id: str):
    """Tests that suite step environments run without errors using random actions."""
    run_env(env_id)


@pytest.mark.parametrize('env_id', SUITE_IDS)
def test_step_suite_determinism(env_id: str):
    """Tests that for step environments identical seeds produce identical trajectories."""
    seed = 0
    run_env_determinism(env_id, seed)


@pytest.mark.parametrize('env_id', MANIPULATION_IDS)
def test_step_manipulation_functionality(env_id: str):
    """Tests that manipulation step environments run without errors using random actions."""
    run_env(env_id)


@pytest.mark.parametrize('env_id', MANIPULATION_IDS)
def test_step_manipulation_determinism(env_id: str):
    """Tests that for step environments identical seeds produce identical trajectories."""
    seed = 0
    run_env_determinism(env_id, seed)


@pytest.mark.parametrize('env_id', [fancy_gym.ALL_DMC_MOVEMENT_PRIMITIVE_ENVIRONMENTS.values()])
def test_bb_dmc_functionality(env_id: str):
    """Tests that black box environments run without errors using random actions."""
    run_env(env_id)


@pytest.mark.parametrize('env_id', [fancy_gym.ALL_DMC_MOVEMENT_PRIMITIVE_ENVIRONMENTS.values()])
def test_bb_dmc_determinism(env_id: str):
    """Tests that for black box environment identical seeds produce identical trajectories."""
    run_env_determinism(env_id)
