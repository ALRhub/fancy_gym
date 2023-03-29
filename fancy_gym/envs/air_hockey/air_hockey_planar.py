import numpy as np
from fancy_gym.envs.air_hockey.air_hockey import AirHockeyBase

from air_hockey_challenge.utils import robot_to_world
from air_hockey_challenge.framework import AirHockeyChallengeWrapper
from air_hockey_challenge.environments.planar import AirHockeyHit, AirHockeyDefend

MAX_EPISODE_STEPS_AIR_HOCKEY_PLANAR_HIT = 120  # default is 500, recommended 120
MAX_EPISODE_STEPS_AIR_HOCKEY_PLANAR_Defend = 180  # default is 500, recommended 180


class AirHockeyPlanarHit(AirHockeyBase):
    def __init__(self):
        super().__init__(env_id="3dof-hit", reward_function=self.planar_hit_reward)

        self.horizon = MAX_EPISODE_STEPS_AIR_HOCKEY_PLANAR_HIT

    @staticmethod
    def planar_hit_reward(base_env: AirHockeyHit, obs, act, obs_, done):
        # init reward and env info
        rew = 0
        env_info = base_env.env_info

        # compute puck_pos, goal_pos and side_pos
        puck_pos, puck_vel = base_env.get_puck(obs_)  # puck_pos, puck_vel in world frame
        goal_pos = np.array([env_info["table"]["length"] / 2, 0, 0])  # goal_pos in world frame
        e_w = env_info["table"]["width"] / 2 - env_info["puck"]["radius"]
        w = (abs(puck_pos[1]) * goal_pos[0] + puck_pos[0] * goal_pos[1] - e_w * puck_pos[0] - e_w * goal_pos[0])
        w = w / (abs(puck_pos[1]) + goal_pos[1] - 2 * e_w)
        side_pos = np.array([w, np.copysign(e_w, puck_pos[1]), 0])  # side_pos in world frame
        if done:
            # if puck in the opponent goal
            x_dis = puck_pos[0] - goal_pos[0]
            y_dis = np.abs(puck_pos[1]) - env_info["table"]["goal_width"] / 2
            if x_dis > 0 > y_dis:
                rew = 200
        else:
            if not base_env.has_hit:
                ee_pos, _ = base_env.get_ee()  # ee_pos, ee_vel in world frame
                ee_puck_dis = np.linalg.norm(ee_pos[:2] - puck_pos[:2])  # distance between ee and puck
                ee_puck_vec = (ee_pos[:2] - puck_pos[:2]) / ee_puck_dis  # vector from ee to puck

                # compute cos between ee_puck and puck_goal
                puck_goal_dis = np.linalg.norm(puck_pos[:2] - goal_pos[:2])  # distance between puck and goal
                puck_goal_vec = (puck_pos[:2] - goal_pos[:2]) / puck_goal_dis  # vector from puck and goal
                cos_ang_goal = np.clip(ee_puck_vec @ puck_goal_vec, 0, 1)  # cos between ee_puck and puck_goal

                # compute cos between ee_puck and puck_side
                puck_side_dis = np.linalg.norm(puck_pos[:2] - side_pos[:2])  # distance between puck and bouncing point
                puck_side_vec = (puck_pos[:2] - side_pos[:2]) / puck_side_dis  # vector from puck to bouncing point
                cos_ang_side = np.clip(ee_puck_vec @ puck_side_vec, 0, 1)  # cos between ee_puck and puck_side

                cos_ang = np.max([cos_ang_goal, cos_ang_side])
                rew = np.exp(-8 * (ee_puck_dis - 0.08)) * cos_ang**2
            else:
                rew_hit = 0.25 + min([1, (0.25 * puck_vel[0] ** 4)])
                rew_goal = 0
                if puck_pos[0] > 0.8:
                    sig = 0.1
                    rew_goal = 1. / (np.sqrt(2. * np.pi) * sig) * np.exp(-np.power((puck_pos[1] - 0) / sig, 2.) / 2)

                # distance
                stay_pos = np.array([-0.5, 0, 0])
                ee_pos, _ = base_env.get_ee()  # ee_pos, ee_vel in world frame
                stay_ee_dist = np.linalg.norm(stay_pos - ee_pos)
                rew_stay = np.exp(-4 * (stay_ee_dist - 0.08))
                # print("hit: ", 2 * rew_hit, "rew_goal: ", 4 * rew_goal, "rew_stay: ", 8 * rew_stay)
                rew = 2 * rew_hit + 4 * rew_goal + 8 * rew_stay

        rew -= 1e-3 * np.linalg.norm(act)
        return rew


class AirHockeyPlanarDefend(AirHockeyBase):
    def __init__(self):
        super().__init__(env_id="3dof-defend", reward_function=self.planar_defend_reward)

        self.horizon = MAX_EPISODE_STEPS_AIR_HOCKEY_PLANAR_Defend

    @ staticmethod
    def planar_defend_reward(base_env: AirHockeyDefend, obs, act, obs_, done):
        # init reward and env info
        rew = 0
        env_info = base_env.env_info

        # compute puck_pos
        puck_pos, puck_vel = base_env.get_puck(obs_)  # puck_pos, puck_vel in world frame
        goal_pos = np.array([-env_info["table"]["length"] / 2, 0, 0])  # self goal_pos in world frame
        if done:
            x_dis = puck_pos[0] - goal_pos[0]
            y_dis = np.abs(puck_pos[1]) - env_info["table"]["goal_width"] / 2
            if x_dis < 0 and y_dis < 0:
                rew = -100
        else:
            if base_env.has_bounce:
                rew = -1
            elif base_env.has_hit:
                if -0.8 < puck_pos[0] < -0.4:
                    r_x = np.exp(-5 * np.abs(puck_pos[0] + 0.6))
                    r_y = 3 * np.exp(-3 * np.abs(puck_pos[1]))
                    r_v = 5 * np.exp(-(5 * np.linalg.norm(puck_vel))**2)
                    rew = r_x + r_y + r_v + 1
            else:
                ee_pos, _ = base_env.get_ee()  # ee_pos, ee_vel in world frame
                ee_pos[0] = 0.5
                ee_puck_dist = np.abs(ee_pos[:2], puck_pos[:2])
                sig = 0.2
                r_x = np.exp(-3 * ee_puck_dist[0])
                r_y = 1. / (np.sqrt(2. * np.pi) * sig) * np.exp(-np.power((ee_puck_dist[1] - 0.08) / sig, 2.) / 2)
                rew = 0.3 * r_x + 0.7 * (r_y/2)
        rew -= 1e-3 * np.linalg.norm(act)
        return rew



