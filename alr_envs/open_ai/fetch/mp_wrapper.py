from typing import Union

from gym import spaces
from mp_env_api.env_wrappers.mp_env_wrapper import MPEnvWrapper


class MPWrapper(MPEnvWrapper):
    @property
    def start_pos(self):
        return self.initial_gripper_xpos

    @property
    def goal_pos(self):
        raise ValueError("Goal position is not available and has to be learnt based on the environment.")

    @property
    def dt(self) -> Union[float, int]:
        return self.env.dt