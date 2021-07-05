from typing import Union

from mp_env_api.env_wrappers.mp_env_wrapper import MPEnvWrapper


class MPWrapper(MPEnvWrapper):
    @property
    def start_pos(self):
        raise ValueError("Start position is not available")

    @property
    def goal_pos(self):
        raise ValueError("Goal position is not available and has to be learnt based on the environment.")

    @property
    def dt(self) -> Union[float, int]:
        return 0.02