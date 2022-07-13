from typing import Union, Tuple

import numpy as np

from alr_envs.black_box.raw_interface_wrapper import RawInterfaceWrapper


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
        return self.env.data.qpos[0:7].copy()

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.env.data.qvel[0:7].copy()

    # TODO: Fix this
    def episode_callback(self, action: np.ndarray, mp) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
        if mp.learn_tau:
            self.env.env.release_step = action[0] / self.env.dt  # Tau value
            return action, None
        else:
            return action, None

    def set_context(self, context):
        xyz = np.zeros(3)
        xyz[:2] = context
        xyz[-1] = 0.840
        self.env.env.model.body_pos[self.env.env.cup_table_id] = xyz
        return self.get_observation_from_step(self.env.env._get_obs())
