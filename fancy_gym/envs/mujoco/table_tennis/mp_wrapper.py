from typing import Union, Tuple

import numpy as np

from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper


class TT_MPWrapper(RawInterfaceWrapper):

    # Random x goal + random init pos
    @property
    def context_mask(self):
        return np.hstack([
            [False] * 7,  # joints position
            [False] * 7,  # joints velocity
            [True] * 2,  # position ball x, y
            [False] * 1,  # position ball z
            #[True] * 3,    # velocity ball x, y, z
            [True] * 2,  # target landing position
            # [True] * 1,  # time
        ])

    @property
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.data.qpos[:7].copy()

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.data.qvel[:7].copy()

    # def preprocessing_and_validity_callback(self, action, pos_traj, vel_traj):
    #     return self.check_traj_validity(action, pos_traj, vel_traj)

    def preprocessing_and_validity_callback(self, action: np.ndarray, pos_traj: np.ndarray, vel_traj: np.ndarray,
                                            tau_bound: list, delay_bound: list):
        return self.check_traj_validity(action, pos_traj, vel_traj, tau_bound, delay_bound)

    def set_episode_arguments(self, action, pos_traj, vel_traj):
        return pos_traj, vel_traj

    # def invalid_traj_callback(self, action: np.ndarray, pos_traj: np.ndarray, vel_traj: np.ndarray,
    #                           return_contextual_obs: bool) -> Tuple[np.ndarray, float, bool, dict]:
    #     return self.get_invalid_traj_step_return(action, pos_traj, return_contextual_obs)

    def invalid_traj_callback(self, action: np.ndarray, pos_traj: np.ndarray, vel_traj: np.ndarray,
                              return_contextual_obs: bool, tau_bound:list, delay_bound:list) -> Tuple[np.ndarray, float, bool, dict]:
        return self.get_invalid_traj_step_return(action, pos_traj, return_contextual_obs, tau_bound, delay_bound)

class TTVelObs_MPWrapper(TT_MPWrapper):

    @property
    def context_mask(self):
        return np.hstack([
            [False] * 7,  # joints position
            [False] * 7,  # joints velocity
            [True] * 2,  # position ball x, y
            [False] * 1,  # position ball z
            [True] * 3,    # velocity ball x, y, z
            [True] * 2,  # target landing position
            # [True] * 1,  # time
        ])