import os
import copy
import random
import numpy as np
from typing import Union, Tuple, Optional

import gym
from gym import spaces, utils
from gym.core import ObsType, ActType

from air_hockey_challenge.framework import AirHockeyChallengeWrapper
from air_hockey_challenge.environments.planar import AirHockeyHit, AirHockeyDefend

MAX_EPISODE_STEPS_AIR_HOCKEY = 150
# MAX_EPISODE_STEPS_AIR_HOCKEY_PLANAR_HIT = 120  # default is 500, recommended 120
# MAX_EPISODE_STEPS_AIR_HOCKEY_PLANAR_Defend = 180  # default is 500, recommended 180


class AirHockeyBase(gym.Env):
    """
    Base Environment for Air Hockey Challenge 2023
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, env_id, reward_function, **kwargs):
        super().__init__()
        # base environment
        self.env = AirHockeyChallengeWrapper(env=env_id, action_type="position-velocity",
                                             interpolation_order=3,
                                             custom_reward_function=reward_function, **kwargs)
        self.base_env = self.env.base_env

        # air hockey env info
        self.mdp_info = self.env.info
        self.env_info = self.env.env_info

        # dt 0.02 or 0.001 ?
        self.dt = self.env_info["dt"]
        self.dof = self.env_info["robot"]["n_joints"]

        # mujoco model and data
        self._model = self.env.base_env._model
        self._data = self.env.base_env._data
        self.robot_model = self.env_info["robot"]["robot_model"]
        self.robot_data = self.env_info["robot"]["robot_data"]

        # observation space
        obs_low = copy.deepcopy(self.mdp_info.observation_space.low)
        obs_high = copy.deepcopy(self.mdp_info.observation_space.high)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # action space
        j_pos_limit = copy.deepcopy(self.env_info["robot"]["joint_pos_limit"])
        j_vel_limit = copy.deepcopy(self.env_info["robot"]["joint_vel_limit"])
        act_low = np.hstack([j_pos_limit[0], j_vel_limit[0]])
        act_high = np.hstack([j_pos_limit[1], j_vel_limit[1]])
        self.action_space = spaces.Box(low=act_low, high=act_high, dtype=np.float32)

        # step counter
        self._episode_steps = 0

        # max steps
        # self.horizon = self.mdp_info.horizon
        self.horizon = MAX_EPISODE_STEPS_AIR_HOCKEY

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        self._episode_steps = 0

        obs = np.array(self.env.reset(), dtype=np.float32)
        info = {}
        if return_info:
            return obs, info
        else:
            return obs

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        act = np.reshape(action, [2, -1])
        obs, rew, done, info = self.env.step(act)
        obs = np.array(obs, dtype=np.float32)

        self._episode_steps += 1
        if self._episode_steps >= self.horizon:
            done = True

        if self.env.base_env.n_agents == 1:
            info["has_hit"] = 1 if self.env.base_env.has_hit else 0

        return obs, rew, done, info

    def render(self, mode="human"):
        if mode == "human":
            self.env.base_env.render(mode="human")
        elif mode == "rgb_array":
            return self.env.base_env.render(mode="rgb_array")
        else:
            raise NotImplementedError

    def seed(self, seed=None):
        self.env.seed(seed)


if __name__ == "__main__":
    pass

