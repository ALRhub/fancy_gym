from typing import Union, Tuple

import numpy as np

from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper


class MiniGolfMPWrapper(RawInterfaceWrapper):
    @property
    def context_mask(self):
        return np.hstack([
            [False] * 7,  # joints position
            [False] * 7,  # joints velocity
            [False] * 3,  # position of ball
            [True] * 3,  # position of red obstacle
            [True] * 3,  # position of green obstacle
            [True] * 1,  # Current width of the target wall
        ])

    @property
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.data.qpos[:7].copy()

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.data.qvel[:7].copy()

    def preprocessing_and_validity_callback(self, action: np.ndarray, pos_traj: np.ndarray, vel_traj: np.ndarray,
                                            tau_bound: list = None, delay_bound: list = None):
        return self.check_traj_validity(action, pos_traj, vel_traj, tau_bound, delay_bound)

    def invalid_traj_callback(self, action: np.ndarray, pos_traj: np.ndarray, vel_traj: np.ndarray,
                              return_contextual_obs: bool, tau_bound:list, delay_bound:list) -> Tuple[np.ndarray, float, bool, dict]:
        return self.get_invalid_traj_step_return(action, pos_traj, return_contextual_obs, tau_bound, delay_bound)


class MiniGolfOneObsMPWrapper(MiniGolfMPWrapper):
    @property
    def context_mask(self):
        return np.hstack([
            [False] * 7,  # joints position
            [False] * 7,  # joints velocity
            [False] * 3,  # position of ball
            [True] * 3,  # position of green obstacle
            [True] * 1,  # Current width of the target wall
        ])