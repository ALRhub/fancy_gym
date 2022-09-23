import fancy_gym
import gym
import pytest

from test.utils import run_env

ALL_SPECS = list(spec for spec in gym.envs.registry.all() if "alr_envs" in spec.entry_point)
SEED = 1


@pytest.mark.parametrize('env_id', fancy_gym.ALL_FANCY_MOVEMENT_PRIMITIVE_ENVIRONMENTS['DMP'])
def test_custom_dmp_functionality(env_id):
    """Tests that environments runs without errors using random actions for custom DMP envs."""
    run_env(env_id)


@pytest.mark.parametrize('env_id', fancy_gym.ALL_FANCY_MOVEMENT_PRIMITIVE_ENVIRONMENTS['ProMP'])
def test_custom_promp_functionality(env_id):
    """Tests that environments runs without errors using random actions for custom ProMP envs."""
    run_env(env_id)


def test_openai_environment_functionality(self):
    """Tests that environments runs without errors using random actions for OpenAI gym MP envs."""
    with self.subTest(msg="DMP"):
        for env_id in alr_envs.ALL_GYM_MOTION_PRIMITIVE_ENVIRONMENTS['DMP']:
            with self.subTest(msg=env_id):
                self.run_env(env_id)

    with self.subTest(msg="ProMP"):
        for env_id in alr_envs.ALL_GYM_MOTION_PRIMITIVE_ENVIRONMENTS['ProMP']:
            with self.subTest(msg=env_id):
                self.run_env(env_id)


def test_dmc_environment_functionality(self):
    """Tests that environments runs without errors using random actions for DMC MP envs."""
    with self.subTest(msg="DMP"):
        for env_id in alr_envs.ALL_DEEPMIND_MOTION_PRIMITIVE_ENVIRONMENTS['DMP']:
            with self.subTest(msg=env_id):
                self.run_env(env_id)

    with self.subTest(msg="ProMP"):
        for env_id in alr_envs.ALL_DEEPMIND_MOTION_PRIMITIVE_ENVIRONMENTS['ProMP']:
            with self.subTest(msg=env_id):
                self.run_env(env_id)


def test_metaworld_environment_functionality(self):
    """Tests that environments runs without errors using random actions for Metaworld MP envs."""
    with self.subTest(msg="DMP"):
        for env_id in alr_envs.ALL_METAWORLD_MOTION_PRIMITIVE_ENVIRONMENTS['DMP']:
            with self.subTest(msg=env_id):
                self.run_env(env_id)

    with self.subTest(msg="ProMP"):
        for env_id in alr_envs.ALL_METAWORLD_MOTION_PRIMITIVE_ENVIRONMENTS['ProMP']:
            with self.subTest(msg=env_id):
                self.run_env(env_id)


def test_alr_environment_determinism(self):
    """Tests that identical seeds produce identical trajectories for ALR MP Envs."""
    with self.subTest(msg="DMP"):
        self._run_env_determinism(alr_envs.ALL_ALR_MOTION_PRIMITIVE_ENVIRONMENTS["DMP"])
    with self.subTest(msg="ProMP"):
        self._run_env_determinism(alr_envs.ALL_ALR_MOTION_PRIMITIVE_ENVIRONMENTS["ProMP"])


def test_openai_environment_determinism(self):
    """Tests that identical seeds produce identical trajectories for OpenAI gym MP Envs."""
    with self.subTest(msg="DMP"):
        self._run_env_determinism(alr_envs.ALL_GYM_MOTION_PRIMITIVE_ENVIRONMENTS["DMP"])
    with self.subTest(msg="ProMP"):
        self._run_env_determinism(alr_envs.ALL_GYM_MOTION_PRIMITIVE_ENVIRONMENTS["ProMP"])


def test_dmc_environment_determinism(self):
    """Tests that identical seeds produce identical trajectories for DMC MP Envs."""
    with self.subTest(msg="DMP"):
        self._run_env_determinism(alr_envs.ALL_DEEPMIND_MOTION_PRIMITIVE_ENVIRONMENTS["DMP"])
    with self.subTest(msg="ProMP"):
        self._run_env_determinism(alr_envs.ALL_DEEPMIND_MOTION_PRIMITIVE_ENVIRONMENTS["ProMP"])


def test_metaworld_environment_determinism(self):
    """Tests that identical seeds produce identical trajectories for Metaworld MP Envs."""
    with self.subTest(msg="DMP"):
        self._run_env_determinism(alr_envs.ALL_METAWORLD_MOTION_PRIMITIVE_ENVIRONMENTS["DMP"])
    with self.subTest(msg="ProMP"):
        self._run_env_determinism(alr_envs.ALL_METAWORLD_MOTION_PRIMITIVE_ENVIRONMENTS["ProMP"])
