from typing import Tuple, Union

import numpy as np

from mp_env_api.envs.positional_env_wrapper import PositionalEnvWrapper


class BallInACupPositionalWrapper(PositionalEnvWrapper):
    @property
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.sim.data.qpos[0:7].copy()

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.sim.data.qvel[0:7].copy()
