from typing import Union, Tuple

import numpy as np

from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper


class MPWrapper(RawInterfaceWrapper):

    # Random x goal + random init pos
    @property
    def context_mask(self):
        return np.hstack([
            [False] * (2 + int(not self.exclude_current_positions_from_observation)),  # position
            [True] * 3,  # set to true if randomize initial pos
            [False] * 6,  # velocity
            [False] * 3,  # goal distance
            [True]  # goal
        ])

    @property
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.data.qpos[3:6].copy()

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.data.qvel[3:6].copy()


    def preprocessing_and_validity_callback(self, action: np.ndarray, pos_traj: np.ndarray, vel_traj: np.ndarray,
                                            tau_bound: list, delay_bound: list):
        return self.check_traj_validity(action, pos_traj, vel_traj, tau_bound, delay_bound)

    def invalid_traj_callback(self, action: np.ndarray, pos_traj: np.ndarray, vel_traj: np.ndarray,
                              return_contextual_obs: bool, tau_bound:list, delay_bound:list) -> Tuple[np.ndarray, float, bool, dict]:
        return self.get_invalid_tau_return(action, pos_traj, return_contextual_obs, tau_bound, delay_bound)