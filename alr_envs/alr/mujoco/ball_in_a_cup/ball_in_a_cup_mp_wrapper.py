from typing import Tuple, Union

import numpy as np

from alr_envs.black_box.raw_interface_wrapper import RawInterfaceWrapper


class BallInACupMPWrapper(RawInterfaceWrapper):

    @property
    def context_mask(self) -> np.ndarray:
        # TODO: @Max Filter observations correctly
        return np.hstack([
            [False] * 7,  # cos
            [False] * 7,  # sin
            # [True] * 2,  # x-y coordinates of target distance
            [False]  # env steps
        ])

    @property
    def start_pos(self):
        if self.simplified:
            return self._start_pos[1::2]
        else:
            return self._start_pos

    @property
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.sim.data.qpos[0:7].copy()

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.sim.data.qvel[0:7].copy()

    @property
    def goal_pos(self):
        # TODO: @Max I think the default value of returning to the start is reasonable here
        raise ValueError("Goal position is not available and has to be learnt based on the environment.")

    @property
    def dt(self) -> Union[float, int]:
        return self.env.dt
