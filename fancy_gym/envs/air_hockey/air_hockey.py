import os
import copy
import random
import numpy as np
from typing import Union, Tuple, Optional

import gym
from gym import spaces, utils
from gym.core import ObsType, ActType

from air_hockey_challenge.framework import AirHockeyChallengeWrapper


MAX_EPISODE_STEPS_AIR_HOCKEY = 200


class AirHockeyChallenge(gym.Env):
    """
    Base Environment for Air Hockey Challenge 2023
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, env_id, custom_reward_function=None, **kwargs):
        super().__init__()
        self.ahc_env = AirHockeyChallengeWrapper(env=env_id, action_type="position-velocity",
                                                 interpolation_order=3, custom_reward_function=None, **kwargs)

        obs_info = self.ahc_env._mdp_info.observation_space
        obs_low = copy.deepcopy(obs_info.low)
        obs_hig = copy.deepcopy(obs_info.high)
        self.observation_space = spaces.Box(low=obs_low, high=obs_hig, dtype=np.float32)

        # act_info = self.ahc_env._mdp_info.action_space
        # act_hig = copy.deepcopy(act_info.high)
        # act_low = copy.deepcopy(act_info.low)
        env_info = self.ahc_env.env_info
        j_pos_limit = copy.deepcopy(env_info["robot"]["joint_pos_limit"])
        j_vel_limit = copy.deepcopy(env_info["robot"]["joint_vel_limit"])
        act_low = np.hstack([j_pos_limit[0], j_vel_limit[0]])
        act_hig = np.hstack([j_pos_limit[1], j_vel_limit[1]])
        self.action_space = spaces.Box(low=act_low, high=act_hig, dtype=np.float32)

        self.step_num = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        self.step_num = 0
        obs = self.ahc_env.reset()
        info = {}
        if return_info:
            return np.array(obs, dtype=np.float32), info
        else:
            return np.array(obs, dtype=np.float32)

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        act = np.reshape(action, [2, -1])
        obs, rew, don, info = self.ahc_env.step(act)

        self.step_num += 1
        if self.step_num == MAX_EPISODE_STEPS_AIR_HOCKEY:
            don = True
        return np.array(obs, dtype=np.float32), rew, don, info

    def render(self, mode="human"):
        self.ahc_env.render()

    def seed(self, seed=None):
        self.ahc_env.seed(seed)


def test_env(env_id: str, seed=0):
    env = AirHockeyChallenge(env_id)
    # env = AirHockeyChallengeWrapper(env_id)
    env.seed(seed)

    os = env.observation_space
    for i in range(5):
        obs = env.reset()
        print(env.observation_space.contains(obs))
        print(obs.dtype)
        print(os.dtype)
        print(np.can_cast(obs.dtype, os.dtype))
        print(obs.shape == os.shape)
        print(np.all(obs >= os.low))
        print(np.all(obs <= os.high))
        stp = 0
        while True:
            act = env.action_space.sample()
            obs_, reward, done, info = env.step(act)
            # env.render()

            stp += 1
            # print(obs_)
            if done:
                # print(stp)
                break


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    test_env("3dof-hit", 0)
    # env = AirHockeyChallengeWrapper("3dof-hit")
    # print(env.env_info)

