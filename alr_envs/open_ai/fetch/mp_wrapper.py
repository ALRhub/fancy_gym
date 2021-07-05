from typing import Union

import numpy as np
from mp_env_api.env_wrappers.mp_env_wrapper import MPEnvWrapper


class MPWrapper(MPEnvWrapper):
    @property
    def current_vel(self) -> Union[float, int, np.ndarray]:
        return self.unwrapped._get_obs()["observation"][-5:-1]

    @property
    def current_pos(self) -> Union[float, int, np.ndarray]:
        return self.unwrapped._get_obs()["observation"][:4]

    @property
    def goal_pos(self):
        raise ValueError("Goal position is not available and has to be learnt based on the environment.")

    @property
    def dt(self) -> Union[float, int]:
        return self.env.dt