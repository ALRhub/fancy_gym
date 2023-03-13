from typing import Tuple, Union

import numpy as np

from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper


class BaseMetaworldMPWrapper(RawInterfaceWrapper):
    @property
    def current_pos(self) -> Union[float, int, np.ndarray]:
        r_close = self.env.data.get_joint_qpos("r_close")
        return np.hstack([self.env.data.mocap_pos.flatten() / self.env.action_scale, r_close])

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return np.zeros(4, )
        # raise NotImplementedError("Velocity cannot be retrieved.")
