from typing import Tuple, Union

import numpy as np

from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper


class MPWrapper(RawInterfaceWrapper):

    def __init__(self, env, n_poles: int = 1):
        self.n_poles = n_poles
        super().__init__(env)

    @property
    def context_mask(self) -> np.ndarray:
        # Besides the ball position, the environment is always set to 0.
        return np.hstack([
            [True],  # slider position
            [True] * 2 * self.n_poles,  # sin/cos hinge angles
            [True],  # slider velocity
            [True] * self.n_poles,  # hinge velocities
        ])

    @property
    def current_pos(self) -> Union[float, int, np.ndarray]:
        return self.env.physics.named.data.qpos["slider"]

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.env.physics.named.data.qvel["slider"]

    @property
    def goal_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        raise ValueError("Goal position is not available and has to be learnt based on the environment.")

    @property
    def dt(self) -> Union[float, int]:
        return self.env.dt


class TwoPolesMPWrapper(MPWrapper):

    def __init__(self, env):
        super().__init__(env, n_poles=2)


class ThreePolesMPWrapper(MPWrapper):

    def __init__(self, env):
        super().__init__(env, n_poles=3)
