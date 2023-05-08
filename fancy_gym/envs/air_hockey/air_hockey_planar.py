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


class AirHockeyPlanarHit(AirHockeyBase):
    def __init__(self, sparse_reward=False):
        if sparse_reward:
            super().__init__(env_id="3dof-hit", reward_function=self.planar_hit_sparse_reward)
        else:
            super().__init__(env_id="3dof-hit", reward_function=self.planar_hit_reward)

        obs_dim = 12
        obs_low = np.ones(obs_dim) * -10000
        obs_high = np.ones(obs_dim) * 10000
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.dt = 0.001
        self.horizon = MAX_EPISODE_STEPS_AIR_HOCKEY_PLANAR_HIT

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        obs, rew, done, info = super().step(action)

        if self._episode_steps >= self.horizon:
            done = True
        else:
            done = False

        if self.env.base_env.n_agents == 1:
            info["has_hit"] = 1 if self.env.base_env.has_hit else 0
            info["has_bounce"] = 1 if self.env.base_env.has_bounce else 0
            info["has_success"] = 1 if self.env.base_env.has_success else 0
            info["hit_step"] = self.env.base_env.hit_step
            info["bounce_step"] = self.env.base_env.bounce_step
            info["success_step"] = self.env.base_env.success_step
            info["jerk_violation"] = np.any(info['jerk'] > 1e4).astype(int)
            info["constr_j_pos"] = np.any(info['constraints_value']['joint_pos_constr'] > 0).astype(int)
            info["constr_j_vel"] = np.any(info['constraints_value']['joint_vel_constr'] > 0).astype(int)
            info["constr_ee"] = np.any(info['constraints_value']['ee_constr'] > 0).astype(int)
            info["validity"] = 1

        return obs, rew, done, info

    @staticmethod
    def check_traj_validity(action, pos_traj, vel_traj):
        invalid_tau = False
        if action.shape[0] % 3 != 0:
            tau_bound = [1.5, 3.0]
            invalid_tau = action[0] < tau_bound[0] or action[0] > tau_bound[1]
        constr_j_pos = np.array([[-2.8, +2.8], [-1.8, +1.8], [-2.0, +2.0]])
        constr_j_vel = np.array([[-1.5, +1.5], [-1.5, +1.5], [-2.0, +2.0]])
        invalid_j_pos = np.any(pos_traj < constr_j_pos[:, 0]) or np.any(pos_traj > constr_j_pos[:, 1])
        invalid_j_vel = np.any(vel_traj < constr_j_vel[:, 0]) or np.any(vel_traj > constr_j_vel[:, 1])
        if invalid_tau or invalid_j_pos or invalid_j_vel:
            return False, pos_traj, vel_traj
        return True, pos_traj, vel_traj

    def _get_invalid_traj_penalty(self, action, traj_pos, traj_vel):
        violate_tau_bound_error = 0
        if action.shape[0] % 3 != 0:
            tau_bound = [1.5, 3.0]
            violate_tau_bound_error = np.max([0, action[0] - tau_bound[1]]) + \
                                      np.max([0, tau_bound[0] - action[0]])
        constr_j_pos = np.array([[-2.8, +2.8], [-1.8, +1.8], [-2.0, +2.0]])
        constr_j_vel = np.array([[-1.5, +1.5], [-1.5, +1.5], [-2.0, +2.0]])
        violate_low_bound_error = np.mean(np.maximum(constr_j_pos[:, 0] - traj_pos, 0)) + \
                                  np.mean(np.maximum(constr_j_vel[:, 0] - traj_vel, 0))
        violate_high_bound_error = np.mean(np.maximum(traj_pos - constr_j_pos[:, 1], 0)) + \
                                   np.mean(np.maximum(traj_vel - constr_j_vel[:, 1], 0))
        invalid_penalty = violate_tau_bound_error + violate_low_bound_error + violate_high_bound_error
        return -invalid_penalty

    def get_invalid_traj_return(self, action, traj_pos, traj_vel):
        obs, rew, done, info = self.step(np.hstack([traj_pos[0], traj_vel[0]]))

        info["jerk_violation"] = 1
        info["constr_j_pos"] = 1
        info["constr_j_vel"] = 1
        info["constr_ee"] = 1
        info["validity"] = 0

        for k, v in info.items():
            info[k] = [v] * self.horizon

        info['trajectory_length'] = self.horizon

        return obs, self._get_invalid_traj_penalty(action, traj_pos, traj_vel), True, info

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

        # if puck in the opponent goal
        x_dis = puck_pos[0] - goal_pos[0]
        y_dis = np.abs(puck_pos[1]) - env_info["table"]["goal_width"] / 2
        has_goal = False
        if x_dis > 0 > y_dis:
            has_goal = True
        if has_goal:
            # distance
            stay_pos = np.array([-0.5, 0, 0])
            ee_pos, _ = base_env.get_ee()  # ee_pos, ee_vel in world frame
            stay_ee_dist = np.linalg.norm(stay_pos - ee_pos)
            rew_stay = np.exp(-4 * (stay_ee_dist - 0.08))
            rew = 20 + 4 * rew_stay
            # print('has_goal', rew)
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
                rew = np.exp(-8 * (ee_puck_dis - 0.08)) * cos_ang ** 2
                # print('not has_hit', rew)
            else:
                rew_hit = 0.25 + min([1, (0.25 * puck_vel[0] ** 4)])
                rew_goal = 0
                if puck_pos[0] > 0.6:
                    sig = 0.1
                    rew_goal = 1. / (np.sqrt(2. * np.pi) * sig) * np.exp(-np.power((puck_pos[1] - 0) / sig, 2.) / 2)

                # distance
                stay_pos = np.array([-0.5, 0, 0])
                ee_pos, _ = base_env.get_ee()  # ee_pos, ee_vel in world frame
                stay_ee_dist = np.linalg.norm(stay_pos - ee_pos)
                rew_stay = np.exp(-4 * (stay_ee_dist - 0.08))
                # print("hit: ", rew_hit, "rew_goal: ", rew_goal, "rew_stay: ", rew_stay)
                rew = 2 * rew_hit + 2 * rew_goal + 4 * rew_stay
                # print('has_hit', rew)

        # print('step', base_env.ep_step)
        rew -= 1e-3 * np.linalg.norm(act)
        return rew

    @staticmethod
    def planar_hit_sparse_reward(base_env: AirHockeyHit, obs, act, obs_, done):
        # init reward and env info
        env_info = base_env.env_info
        goal_pos = np.array([env_info["table"]["length"] / 2, 0, 0])

        # record data
        ee_pos, ee_vel = base_env.get_ee()
        base_env.traj_ee_pos.append(ee_pos)
        base_env.traj_ee_vel.append(ee_vel)
        puck_pos, puck_vel = base_env.get_puck(obs_)
        base_env.traj_puck_pos.append(puck_pos)
        base_env.traj_puck_vel.append(puck_vel)

        # ee to puck
        ee_puck_dis = np.linalg.norm(ee_pos[:2] - puck_pos[:2])  # distance between ee and puck
        ee_puck_vec = (ee_pos[:2] - puck_pos[:2]) / ee_puck_dis  # vector from ee to puck

        # compute cos between ee_puck and puck_goal
        puck_goal_dis = np.linalg.norm(puck_pos[:2] - goal_pos[:2])  # distance between puck and goal
        puck_goal_vec = (puck_pos[:2] - goal_pos[:2]) / puck_goal_dis  # vector from puck and goal
        cos_ang_goal = np.clip(ee_puck_vec @ puck_goal_vec, 0, 1)  # cos between ee_puck and puck_goal
        base_env.traj_cos_ang.append(cos_ang_goal)

        if base_env.ep_step < MAX_EPISODE_STEPS_AIR_HOCKEY_PLANAR_HIT:
            return 0

        traj_ee_pos = np.vstack(base_env.traj_ee_pos)
        traj_ee_vel = np.vstack(base_env.traj_ee_vel)
        traj_puck_pos = np.vstack(base_env.traj_puck_pos)
        traj_puck_vel = np.vstack(base_env.traj_puck_vel)
        traj_cos_ang = np.stack(base_env.traj_cos_ang)

        # ee constr violation
        table_width = env_info['table']['width']
        table_length = env_info['table']['length']
        invalid_ee = np.any(np.abs(traj_ee_pos[:, 1]) > table_width / 2) or \
                     np.any(traj_ee_pos[:, 0] < -table_length / 2)
        # if invalid_ee:
        #     return -1

        # get score
        if base_env.has_success:
            # min_success_step = MAX_EPISODE_STEPS_AIR_HOCKEY_PLANAR_HIT / 2
            # max_success_step = MAX_EPISODE_STEPS_AIR_HOCKEY_PLANAR_HIT / 1
            coef = np.clip(1.0 - (100 - base_env.success_step) / 50, 0, 1)
            success_reward = 6
            return 4 + coef * success_reward

        if base_env.has_bounce:
            pass

        if base_env.has_hit:
            hit_step = base_env.hit_step
            cos_ang = traj_cos_ang[hit_step]
            idx = np.argmin(np.linalg.norm(traj_puck_pos - goal_pos, axis=1))
            min_p_g_dist = np.linalg.norm(traj_puck_pos[idx] - goal_pos)
            max_p_vel_x = np.max(traj_puck_vel[:, 0])
            # p_pos_y = np.abs(traj_puck_pos[idx, 1])
            # p_vel_x = traj_puck_vel[idx, 0] if traj_puck_vel[idx, 0] > 0 else 0
            return 1 + 1.0 * cos_ang + 1.0 * (1 - np.tanh(min_p_g_dist)) + 1.0 * np.tanh(max_p_vel_x)

        idx = np.argmin(np.linalg.norm(traj_puck_pos - traj_ee_pos, axis=1))
        coef = traj_cos_ang[idx]
        min_e_p_dist = np.linalg.norm(traj_puck_pos[idx] - traj_ee_pos[idx])
        return coef * (1 - np.tanh(min_e_p_dist))  # [0, 1]


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
