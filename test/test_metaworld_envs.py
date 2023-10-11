from itertools import chain

import pytest
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

import fancy_gym
from test.utils import run_env, run_env_determinism

METAWORLD_IDS = [f'metaworld/{env.split("-goal-observable")[0]}' for env, _ in
                 ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE.items()]
METAWORLD_MP_IDS = fancy_gym.ALL_METAWORLD_MOVEMENT_PRIMITIVE_ENVIRONMENTS['all']
SEED = 1


@pytest.mark.parametrize('env_id', METAWORLD_IDS)
def test_step_metaworld_functionality(env_id: str):
    """Tests that step environments run without errors using random actions."""
    run_env(env_id)


@pytest.mark.skip(reason="Seeding does not correctly work on current Metaworld.")
@pytest.mark.parametrize('env_id', METAWORLD_IDS)
def test_step_metaworld_determinism(env_id: str):
    """Tests that for step environments identical seeds produce identical trajectories."""
    run_env_determinism(env_id, SEED)


@pytest.mark.parametrize('env_id', METAWORLD_MP_IDS)
def test_bb_metaworld_functionality(env_id: str):
    """Tests that black box environments run without errors using random actions."""
    run_env(env_id)


@pytest.mark.skip(reason="Seeding does not correctly work on current Metaworld.")
@pytest.mark.parametrize('env_id', METAWORLD_MP_IDS)
def test_bb_metaworld_determinism(env_id: str):
    """Tests that for black box environment identical seeds produce identical trajectories."""
    run_env_determinism(env_id, SEED)
