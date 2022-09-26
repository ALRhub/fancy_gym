from typing import Tuple, Union

import numpy as np

from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper


class MPWrapper(RawInterfaceWrapper):

    @property
    def context_mask(self) -> np.ndarray:
        # Joint and target positions are randomized, velocities are always set to 0.
        return np.hstack([
            [True] * 2,  # joint position
            [True] * 2,  # target position
            [False] * 2,  # joint velocity
        ])

    @property
    def current_pos(self) -> Union[float, int, np.ndarray]:
        return self.env.physics.named.data.qpos[:]

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.env.physics.named.data.qvel[:]

    @property
    def goal_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        raise ValueError("Goal position is not available and has to be learnt based on the environment.")

    @property
    def dt(self) -> Union[float, int]:
        return self.env.dt
