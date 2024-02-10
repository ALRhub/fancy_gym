import pytest

from fancy_gym.examples.example_replanning_envs import main as replanning_envs_main
from fancy_gym.examples.examples_dmc import main as dmc_main
from fancy_gym.examples.examples_general import main as general_main
from fancy_gym.examples.examples_metaworld import main as metaworld_main
from fancy_gym.examples.examples_movement_primitives import main as mp_main
from fancy_gym.examples.examples_open_ai import main as open_ai_main

@pytest.mark.parametrize('entry', [replanning_envs_main, dmc_main, general_main, metaworld_main, mp_main, open_ai_main])
@pytest.mark.parametrize('render', [False])
def test_run_example(entry, render):
    entry(render=render)
