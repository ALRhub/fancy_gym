from alr_envs.mp.black_box_wrapper import BlackBoxWrapper
from typing import Union, Tuple
import numpy as np


class MPWrapper(BlackBoxWrapper):
    @property
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.env.sim.data.qpos[3:6].copy()

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.env.sim.data.qvel[3:6].copy()

    # # random goal
    # def set_active_obs(self):
    #     return np.hstack([
    #         [False] * (5 + int(not self.env.exclude_current_positions_from_observation)),  # position
    #         [False] * 6,  # velocity
    #         [True]
    #     ])

    # Random x goal + random init pos
    def get_context_mask(self):
        return np.hstack([
                [False] * (2 + int(not self.env.exclude_current_positions_from_observation)),  # position
                [True] * 3,    # set to true if randomize initial pos
                [False] * 6,  # velocity
                [True]
            ])


class NewHighCtxtMPWrapper(MPWrapper):
    def get_context_mask(self):
        return np.hstack([
            [False] * (2 + int(not self.env.exclude_current_positions_from_observation)),  # position
            [True] * 3,  # set to true if randomize initial pos
            [False] * 6,  # velocity
            [True],     # goal
            [False] * 3 # goal diff
        ])

    def set_context(self, context):
        return self.get_observation_from_step(self.env.env.set_context(context))

