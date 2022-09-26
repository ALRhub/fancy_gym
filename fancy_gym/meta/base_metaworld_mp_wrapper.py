from abc import ABC
from typing import Tuple, Union

import numpy as np

from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper


class BaseMetaworldMPWrapper(RawInterfaceWrapper, ABC):
    @property
    def current_pos(self) -> Union[float, int, np.ndarray]:
        r_close = self.env.data.get_joint_qpos("r_close")
        # TODO check if this is correct
        # return np.hstack([self.env.data.get_body_xpos('hand').flatten() / self.env.action_scale, r_close])
        return np.hstack([self.env.data.mocap_pos.flatten() / self.env.action_scale, r_close])

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        # TODO check if this is correct
        return np.zeros(4, )
        # raise NotImplementedError("Velocity cannot be retrieved.")
