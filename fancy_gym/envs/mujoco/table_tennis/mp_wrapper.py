from typing import Union, Tuple

import numpy as np

from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper
from fancy_gym.envs.mujoco.table_tennis.table_tennis_utils import jnt_pos_low, jnt_pos_high, delay_bound, tau_bound


class MPWrapper(RawInterfaceWrapper):

    # Random x goal + random init pos
    @property
    def context_mask(self):
        return np.hstack([
            [False] * 7,  # joints position
            [False] * 7,  # joints velocity
            [True] * 2,  # position ball x, y
            [False] * 1,  # position ball z
            [True] * 2,  # target landing position
            # [True] * 1,  # time
        ])

    @property
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.data.qpos[:7].copy()

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.data.qvel[:7].copy()

    def episode_callback(self, action, pos_traj, vel_traj):
        time_invalid = action[0] > tau_bound[1] or action[0] < tau_bound[0] \
                     or action[1] > delay_bound[1] or action[1] < delay_bound[0]
        if time_invalid or np.any(pos_traj > jnt_pos_high) or np.any(pos_traj < jnt_pos_low):
            return False
        return True

    def invalid_traj_callback(self, action, pos_traj: np.ndarray, vel_traj: np.ndarray) \
            -> Tuple[np.ndarray, float, bool, dict]:
        tau_invalid_penalty = 3 * (np.max([0, action[0] - tau_bound[1]]) + np.max([0, tau_bound[0] - action[0]]))
        delay_invalid_penalty = 3 * (np.max([0, action[1] - delay_bound[1]]) + np.max([0, delay_bound[0] - action[1]]))
        violate_high_bound_error = np.mean(np.maximum(pos_traj - jnt_pos_high, 0))
        violate_low_bound_error = np.mean(np.maximum(jnt_pos_low - pos_traj, 0))
        invalid_penalty = tau_invalid_penalty + delay_invalid_penalty + \
                          violate_high_bound_error + violate_low_bound_error
        return self.get_obs(), -invalid_penalty, True, {
        "hit_ball": [False],
        "ball_returned_success": [False],
        "land_dist_error": [10.],
        "is_success": [False],
        'trajectory_length': 1,
        "num_steps": [1]
        }