import itertools

import fancy_gym
import gym
import pytest

from test.utils import run_env, run_env_determinism

CUSTOM_IDS = [spec.id for spec in gym.envs.registry.all() if
              "fancy_gym" in spec.entry_point and 'make_bb_env_helper' not in spec.entry_point]
CUSTOM_MP_IDS = itertools.chain(*fancy_gym.ALL_FANCY_MOVEMENT_PRIMITIVE_ENVIRONMENTS.values())
SEED = 1


@pytest.mark.parametrize('env_id', CUSTOM_IDS)
def test_step_fancy_functionality(env_id: str):
    """Tests that step environments run without errors using random actions."""
    run_env(env_id)


@pytest.mark.parametrize('env_id', CUSTOM_IDS)
def test_step_fancy_determinism(env_id: str):
    """Tests that for step environments identical seeds produce identical trajectories."""
    run_env_determinism(env_id, SEED)


@pytest.mark.parametrize('env_id', CUSTOM_MP_IDS)
def test_bb_fancy_functionality(env_id: str):
    """Tests that black box environments run without errors using random actions."""
    run_env(env_id)


@pytest.mark.parametrize('env_id', CUSTOM_MP_IDS)
def test_bb_fancy_determinism(env_id: str):
    """Tests that for black box environment identical seeds produce identical trajectories."""
    run_env_determinism(env_id, SEED)
