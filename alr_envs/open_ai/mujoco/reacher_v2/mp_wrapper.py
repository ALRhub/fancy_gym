from typing import Union

import numpy as np

from alr_envs.black_box.raw_interface_wrapper import RawInterfaceWrapper


class MPWrapper(RawInterfaceWrapper):

    @property
    def current_vel(self) -> Union[float, int, np.ndarray]:
        return self.sim.data.qvel[:2]

    @property
    def current_pos(self) -> Union[float, int, np.ndarray]:
        return self.sim.data.qpos[:2]
