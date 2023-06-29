import copy
import numpy as np
from typing import Union, Tuple, Optional

import gym
from gym import spaces, utils
from gym.core import ObsType, ActType

from air_hockey_challenge.framework import AirHockeyChallengeWrapper

MAX_EPISODE_STEPS_AIR_HOCKEY = 200


class AirHockeyGymBase(gym.Env):
    """
    Base Environment for Air Hockey Challenge 2023
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, env_id, interpolation_order=3, custom_reward_function=None, **kwargs):
        super().__init__()
        # base environment
        self.env = AirHockeyChallengeWrapper(env=env_id,
                                             interpolation_order=interpolation_order,
                                             custom_reward_function=custom_reward_function,
                                             **kwargs)
        self.base_env = self.env.base_env
        self.interpolation_order = interpolation_order

        # air hockey env info
        self.mdp_info = self.env.info
        self.env_info = self.env.env_info

        # dt and dof
        self.dt = self.env_info["dt"]
        self.dof = self.env_info["robot"]["n_joints"]

        # mujoco model and data
        # self._model = self.base_env._model
        # self._data = self.base_env._data
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

        # action related
        self.prev_pos = np.zeros([1, self.dof])
        self.prev_vel = np.zeros([1, self.dof])

    def reset(self, **kwargs) -> Union[ObsType, Tuple[ObsType, dict]]:
        self._episode_steps = 0
        self.prev_pos = np.zeros([1, self.dof])
        self.prev_vel = np.zeros([1, self.dof])
        return np.array(self.env.reset(), dtype=np.float32)

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:

        if self.interpolation_order is None:
            pos = action[:, :self.dof].copy()
            vel = action[:, self.dof:].copy()
            acc = np.diff(vel, axis=0, prepend=self.prev_vel) / self.dt
            self.prev_vel[0] = vel[-1]
            act = np.stack([pos, vel, acc], axis=1)
        else:
            act = action
            if self.interpolation_order == -1:
                act = np.reshape(action, [2, -1])
            if self.interpolation_order == +1:
                act = np.reshape(action, [1, -1])
            if self.interpolation_order == +2:
                act = np.reshape(action, [1, -1])
            if self.interpolation_order == +3:
                act = np.reshape(action, [2, -1])
            if self.interpolation_order == +4:
                act = np.reshape(action, [2, -1])
            if self.interpolation_order == +5:
                act = np.reshape(action, [3, -1])

        obs, rew, done, info = self.env.step(act)

        self._episode_steps += 1
        done = True if self._episode_steps >= self.horizon else done

        return np.array(obs, dtype=np.float32), rew, done, info

    def render(self, mode="human"):
        if mode == "human":
            self.env.base_env.render(mode="human")
        elif mode == "rgb_array":
            return self.env.base_env.render(mode="rgb_array")
        else:
            raise NotImplementedError

    def seed(self, seed=None):
        self.base_env.seed(seed)


if __name__ == "__main__":
    pass

