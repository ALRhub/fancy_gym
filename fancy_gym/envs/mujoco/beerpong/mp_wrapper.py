from typing import Union, Tuple

import numpy as np

from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper


class MPWrapper(RawInterfaceWrapper):

    @property
    def context_mask(self) -> np.ndarray:
        return np.hstack([
            [False] * 7,  # cos
            [False] * 7,  # sin
            [False] * 7,  # joint velocities
            [False] * 3,  # cup_goal_diff_final
            [False] * 3,  # cup_goal_diff_top
            [True] * 2,  # xy position of cup
            # [False]  # env steps
        ])

    @property
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.data.qpos[0:7].copy()

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.data.qvel[0:7].copy()

    # TODO: Fix this
    def episode_callback(self, action: np.ndarray, mp) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
        if mp.learn_tau:
            self.release_step = action[0] / self.dt  # Tau value
        return action, None

    def set_episode_arguments(self, action, pos_traj, vel_traj):
        # self.env.release_step = action[0] / self.dt  # Tau value
        self.env.env.env.env.release_step = action[0] / self.dt  # Tau value
        return pos_traj, vel_traj
    # def preprocessing_and_validity_callback(self, action: np.ndarray, pos_traj: np.ndarray, vel_traj: np.ndarray,
    #                                         tau_bound: list = None, delay_bound: list = None) -> Tuple[bool, np.ndarray, np.ndarray]:
    #     self.release_step = action[0]/ self.dt  # Tau value
    #     return True, pos_traj, vel_traj

    # def set_context(self, context):
    #     xyz = np.zeros(3)
    #     xyz[:2] = context
    #     xyz[-1] = 0.840
    #     self.model.body_pos[self.cup_table_id] = xyz
    #     return self.get_observation_from_step(self.get_obs())
