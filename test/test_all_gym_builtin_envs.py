import re
from itertools import chain
from typing import Callable

import gymnasium as gym
import pytest

import fancy_gym
from test.utils import run_env, run_env_determinism

GYM_IDS = [spec.id for spec in gym.envs.registry.values() if
           not isinstance(spec.entry_point, Callable) and
           "fancy_gym" not in spec.entry_point and 'make_bb_env_helper' not in spec.entry_point
           and 'jax' not in spec.id.lower()
           and 'shimmy' not in spec.id.lower()
           and 'ale_py' not in spec.id.lower()
           and 'tabular' not in spec.id.lower()
           and not re.match(r'GymV2.Environment', spec.id)
           ]
GYM_MP_IDS = fancy_gym.ALL_DMC_MOVEMENT_PRIMITIVE_ENVIRONMENTS['all']
SEED = 1


@pytest.mark.parametrize('env_id', GYM_IDS)
def test_step_gym_functionality(env_id: str):
    """Tests that step environments run without errors using random actions."""
    run_env(env_id)


@pytest.mark.parametrize('env_id', GYM_IDS)
def test_step_gym_determinism(env_id: str):
    """Tests that for step environments identical seeds produce identical trajectories."""
    run_env_determinism(env_id, SEED)


@pytest.mark.parametrize('env_id', GYM_MP_IDS)
def test_bb_gym_functionality(env_id: str):
    """Tests that black box environments run without errors using random actions."""
    run_env(env_id)


@pytest.mark.parametrize('env_id', GYM_MP_IDS)
def test_bb_gym_determinism(env_id: str):
    """Tests that for black box environment identical seeds produce identical trajectories."""
    run_env_determinism(env_id, SEED)
