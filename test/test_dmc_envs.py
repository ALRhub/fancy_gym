from itertools import chain
from typing import Callable

import gymnasium as gym
import pytest

import fancy_gym
from test.utils import run_env, run_env_determinism

DMC_IDS = [spec.id for spec in gym.envs.registry.values() if
           spec.id.startswith('dm_control/')
           and 'compatibility-env-v0' not in spec.id
           and 'lqr-lqr' not in spec.id]
DMC_MP_IDS = fancy_gym.ALL_DMC_MOVEMENT_PRIMITIVE_ENVIRONMENTS['all']
SEED = 1


@pytest.mark.parametrize('env_id', DMC_IDS)
def test_step_dm_control_functionality(env_id: str):
    """Tests that suite step environments run without errors using random actions."""
    run_env(env_id, 5000, wrappers=[gym.wrappers.FlattenObservation])


@pytest.mark.parametrize('env_id', DMC_IDS)
def test_step_dm_control_determinism(env_id: str):
    """Tests that for step environments identical seeds produce identical trajectories."""
    run_env_determinism(env_id, SEED, 5000, wrappers=[gym.wrappers.FlattenObservation])


@pytest.mark.parametrize('env_id', DMC_MP_IDS)
def test_bb_dmc_functionality(env_id: str):
    """Tests that black box environments run without errors using random actions."""
    run_env(env_id)


@pytest.mark.parametrize('env_id', DMC_MP_IDS)
def test_bb_dmc_determinism(env_id: str):
    """Tests that for black box environment identical seeds produce identical trajectories."""
    run_env_determinism(env_id, SEED)
