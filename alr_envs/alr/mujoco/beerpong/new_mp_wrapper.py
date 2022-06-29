from typing import Union, Tuple

import numpy as np

from alr_envs.mp.raw_interface_wrapper import RawInterfaceWrapper


class NewMPWrapper(RawInterfaceWrapper):
    @property
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.env.sim.data.qpos[0:7].copy()

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.env.sim.data.qvel[0:7].copy()

    def get_context_mask(self):
        return np.hstack([
            [False] * 7,  # cos
            [False] * 7,  # sin
            [False] * 7,  # joint velocities
            [False] * 3,  # cup_goal_diff_final
            [False] * 3,  # cup_goal_diff_top
            [True] * 2,  # xy position of cup
            [False]  # env steps
        ])

    def _episode_callback(self, action: np.ndarray) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
        if self.mp.learn_tau:
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
