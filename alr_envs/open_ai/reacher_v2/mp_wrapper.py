from typing import Union

from mp_env_api.env_wrappers.mp_env_wrapper import MPEnvWrapper


class MPWrapper(MPEnvWrapper):

    @property
    def start_pos(self):
        raise ValueError("Start position is not available")

    @property
    def goal_pos(self):
        return self.goal

    @property
    def dt(self) -> Union[float, int]:
        return self.env.dt