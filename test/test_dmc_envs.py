from itertools import chain

import pytest
from dm_control import suite, manipulation

import fancy_gym
from test.utils import run_env, run_env_determinism

SUITE_IDS = [f'dmc:{env}-{task}' for env, task in suite.ALL_TASKS if env != "lqr"]
MANIPULATION_IDS = [f'dmc:manipulation-{task}' for task in manipulation.ALL if task.endswith('_features')]
DMC_MP_IDS = chain(*fancy_gym.ALL_DMC_MOVEMENT_PRIMITIVE_ENVIRONMENTS.values())
SEED = 1


@pytest.mark.parametrize('env_id', SUITE_IDS)
def test_step_suite_functionality(env_id: str):
    """Tests that suite step environments run without errors using random actions."""
    run_env(env_id)


@pytest.mark.parametrize('env_id', SUITE_IDS)
def test_step_suite_determinism(env_id: str):
    """Tests that for step environments identical seeds produce identical trajectories."""
    run_env_determinism(env_id, SEED)


@pytest.mark.parametrize('env_id', MANIPULATION_IDS)
def test_step_manipulation_functionality(env_id: str):
    """Tests that manipulation step environments run without errors using random actions."""
    run_env(env_id)


@pytest.mark.parametrize('env_id', MANIPULATION_IDS)
def test_step_manipulation_determinism(env_id: str):
    """Tests that for step environments identical seeds produce identical trajectories."""
    run_env_determinism(env_id, SEED)


@pytest.mark.parametrize('env_id', DMC_MP_IDS)
def test_bb_dmc_functionality(env_id: str):
    """Tests that black box environments run without errors using random actions."""
    run_env(env_id)


@pytest.mark.parametrize('env_id', DMC_MP_IDS)
def test_bb_dmc_determinism(env_id: str):
    """Tests that for black box environment identical seeds produce identical trajectories."""
    run_env_determinism(env_id, SEED)
