from itertools import chain

import pytest

import fancy_gym
from test.utils import run_env, run_env_determinism

Fancy_ProDMP_IDS = fancy_gym.ALL_FANCY_MOVEMENT_PRIMITIVE_ENVIRONMENTS['ProDMP']

All_ProDMP_IDS = fancy_gym.ALL_MOVEMENT_PRIMITIVE_ENVIRONMENTS['ProDMP']

@pytest.mark.parametrize('env_id', Fancy_ProDMP_IDS)
def test_prodmp_envs(env_id: str):
    """Tests that ProDMP environments run without errors using random actions."""
    run_env(env_id)