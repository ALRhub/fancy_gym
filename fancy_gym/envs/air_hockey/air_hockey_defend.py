from typing import Union, Tuple, Optional

import numpy as np
from gym import spaces, utils
from gym.core import ObsType, ActType
from fancy_gym.envs.air_hockey.air_hockey import AirHockeyBase

from air_hockey_challenge.utils import robot_to_world
from air_hockey_challenge.framework import AirHockeyChallengeWrapper
from air_hockey_challenge.environments.planar import AirHockeyHit, AirHockeyDefend

MAX_EPISODE_STEPS_AIR_HOCKEY_PLANAR_HIT = 150  # default is 500, recommended 120
MAX_EPISODE_STEPS_AIR_HOCKEY_PLANAR_Defend = 180  # default is 500, recommended 180


class AirHockeyPlanarDefend(AirHockeyBase):
    def __init__(self):
        super().__init__(env_id="3dof-defend", reward_function=self.planar_defend_reward)

        self.horizon = MAX_EPISODE_STEPS_AIR_HOCKEY_PLANAR_Defend

    @staticmethod
    def planar_defend_reward(base_env: AirHockeyDefend, obs, act, obs_, done):
        # init reward and env info
        rew = 0
        env_info = base_env.env_info

        # # compute puck_pos
        # puck_pos, puck_vel = base_env.get_puck(obs_)  # puck_pos, puck_vel in world frame
        # goal_pos = np.array([-env_info["table"]["length"] / 2, 0, 0])  # self goal_pos in world frame
        # if done:
        #     x_dis = puck_pos[0] - goal_pos[0]
        #     y_dis = np.abs(puck_pos[1]) - env_info["table"]["goal_width"] / 2
        #     if x_dis < 0 and y_dis < 0:
        #         rew = -100
        # else:
        #     if base_env.has_bounce:
        #         rew = -1
        #     elif base_env.has_hit:
        #         if -0.8 < puck_pos[0] < -0.4:
        #             r_x = np.exp(-5 * np.abs(puck_pos[0] + 0.6))
        #             r_y = 3 * np.exp(-3 * np.abs(puck_pos[1]))
        #             r_v = 5 * np.exp(-(5 * np.linalg.norm(puck_vel))**2)
        #             rew = r_x + r_y + r_v + 1
        #     else:
        #         ee_pos, _ = base_env.get_ee()  # ee_pos, ee_vel in world frame
        #         ee_pos[0] = 0.5
        #         ee_puck_dist = np.abs(ee_pos[:2], puck_pos[:2])
        #         sig = 0.2
        #         r_x = np.exp(-3 * ee_puck_dist[0])
        #         r_y = 1. / (np.sqrt(2. * np.pi) * sig) * np.exp(-np.power((ee_puck_dist[1] - 0.08) / sig, 2.) / 2)
        #         rew = 0.3 * r_x + 0.7 * (r_y/2)
        # rew -= 1e-3 * np.linalg.norm(act)
        return rew
