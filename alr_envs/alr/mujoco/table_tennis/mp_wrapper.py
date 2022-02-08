from typing import Tuple, Union

import numpy as np

from mp_env_api.interface_wrappers.mp_env_wrapper import MPEnvWrapper


class MPWrapper(MPEnvWrapper):

    @property
    def active_obs(self):
        # TODO: @Max Filter observations correctly
        return np.hstack([
            [False] * 7,  # Joint Pos
            [True] * 2, # Ball pos
            [True] * 2  # goal pos
        ])

    @property
    def start_pos(self):
        return self.self.init_qpos_tt

    @property
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.sim.data.qpos[:7].copy()

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.sim.data.qvel[:7].copy()

    @property
    def goal_pos(self):
        # TODO: @Max I think the default value of returning to the start is reasonable here
        raise ValueError("Goal position is not available and has to be learnt based on the environment.")

    @property
    def dt(self) -> Union[float, int]:
        return self.env.dt
