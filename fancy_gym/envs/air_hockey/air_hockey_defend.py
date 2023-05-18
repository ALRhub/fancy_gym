import copy
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
    def __init__(self, reward_function: Union[str, None] = None, invtraj=0):
        super().__init__(env_id="3dof-defend", reward_function=reward_functions[reward_function])

        obs_dim = 12
        obs_low = np.ones(obs_dim) * -10000
        obs_high = np.ones(obs_dim) * 10000
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.dt = 0.001
        self.horizon = MAX_EPISODE_STEPS_AIR_HOCKEY_PLANAR_Defend

        self.invtraj = invtraj

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        obs, rew, done, info = super().step(action)

        if self._episode_steps >= self.horizon:
            done = True
        else:
            done = False

        if self.env.base_env.n_agents == 1:
            info["validity"] = 1
            info["valid_trajectory_reward"] = rew
            info["invalid_trajectory_reward"] = 0

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

    @staticmethod
    def get_invalid_traj_penalty(action, traj_pos, traj_vel):
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

        invalid_rew = self.get_invalid_traj_penalty(action, traj_pos, traj_vel) + self.invtraj

        # in fancy gym added metrics
        info["validity"] = 0
        info["valid_trajectory_reward"] = rew
        info["invalid_trajectory_reward"] = invalid_rew

        # default metrics
        info["has_hit"] = 0
        info["has_success"] = 0
        info["has_hit_step"] = self.horizon
        info["has_success_step"] = self.horizon
        info["current_episode_length"] = self.horizon
        info["max_j_pos_violation"] = 10
        info["max_j_vel_violation"] = 10
        info["max_ee_x_violation"] = 10
        info["max_ee_y_violation"] = 10
        info["max_ee_z_violation"] = 10
        info["max_jerk_violation"] = 10
        info["num_j_pos_violation"] = self.horizon
        info["num_j_vel_violation"] = self.horizon
        info["num_ee_x_violation"] = self.horizon
        info["num_ee_y_violation"] = self.horizon
        info["num_ee_z_violation"] = self.horizon
        info["num_jerk_violation"] = self.horizon

        for k, v in info.items():
            info[k] = [v] * self.horizon

        info['trajectory_length'] = self.horizon

        return obs, invalid_rew, True, info

    @staticmethod
    def planar_defend_original_reward(base_env: AirHockeyDefend, state, action, next_state, absorbing):
        action_penalty = 1e-3

        r = 0
        puck_pos, puck_vel = base_env.get_puck(next_state)
        env_info = base_env.env_info

        # If absorbing the puck is out of bounds of the table.
        if absorbing:
            # large penalty if agent coincides a goal
            if puck_pos[0] + env_info['table']['length'] / 2 < 0 and \
                    np.abs(puck_pos[1]) - env_info['table']['goal_width'] < 0:
                r = -100
        else:
            # If the puck bounced off the head walls, there is no reward.
            if base_env.has_bounce:
                r = -1
            elif base_env.has_hit:
                # Reward if the puck slows down on the defending side
                if -0.8 < puck_pos[0] < -0.4:
                    r_y = 3 * np.exp(-3 * np.abs(puck_pos[1]))
                    r_x = np.exp(-5 * np.abs(puck_pos[0] + 0.6))
                    r_vel = 5 * np.exp(-(5 * np.linalg.norm(puck_vel))**2)
                    r = r_x + r_y + r_vel + 1

                # If we did not yet hit the puck, reward is controlled by the distance between end effector and puck
                # on the x axis
            else:
                ee_pos = base_env.get_ee()[0][:2]

                # Maybe change -0.6 to -0.4 so the puck is stopped a bit higher, could improve performance because
                # we don't run into the constraints at the bottom
                ee_des = np.array([-0.6, puck_pos[1]])
                dist_ee_puck = np.abs(ee_des - ee_pos)

                r_x = np.exp(-3 * dist_ee_puck[0])

                sig = 0.2
                r_y = 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((dist_ee_puck[1] - 0.08)/sig, 2.)/2)
                r = 0.3 * r_x + 0.7 * (r_y/2)

        # penalizes the amount of torque used
        r -= action_penalty * np.linalg.norm(action)
        return r

    @staticmethod
    def planar_defend_sparse_reward1(base_env: AirHockeyDefend, state, action, next_state, absorbing):
        env_info = base_env.env_info


        if base_env.ep_step < MAX_EPISODE_STEPS_AIR_HOCKEY_PLANAR_Defend:
            return 0

        traj_ee_pos = np.vstack(base_env.ee_pos_history)
        traj_ee_vel = np.vstack(base_env.ee_vel_history)
        traj_puck_pos = np.vstack(base_env.puck_pos_history)
        traj_puck_vel = np.vstack(base_env.puck_vel_history)
        traj_cos_ee_puck_goal = np.stack(base_env.cos_ee_puck_goal_history)

        # get score
        if base_env.has_goal:
            return -20

        # we only need to calculate p_yrel if has_bounce
        p_yrel = 0
        if base_env.has_bounce:
            table_width = env_info["table"]["width"]
            goal_width = env_info["table"]["goal_width"]
            bounce_len = (table_width / 2) - (goal_width / 2)

            puck_y = traj_puck_pos[base_env.bounce_step, 1]
            puck_len = np.abs(puck_y) - (goal_width / 2)

            p_yrel = 1 - (puck_len / bounce_len)

        if base_env.has_bounce and base_env.has_hit:
            return -10 * p_yrel + 10

        if base_env.has_bounce:
            return -20 * p_yrel

        if base_env.has_hit:
            return 20

        # print("WARNING Not defined reward, somehow came here! WARNING") # too slow etc
        return 0

    @staticmethod
    def planar_defend_sparse_reward2(base_env: AirHockeyDefend, state, action, next_state, absorbing):
        env_info = base_env.env_info

        # record data
        # ee_pos, ee_vel = base_env.get_ee()
        # base_env.traj_ee_pos.append(ee_pos)
        # base_env.traj_ee_vel.append(ee_vel)
        puck_pos, puck_vel = base_env.get_puck(next_state)
        base_env.traj_puck_pos.append(puck_pos)
        base_env.traj_puck_vel.append(puck_vel)

        if base_env.ep_step < MAX_EPISODE_STEPS_AIR_HOCKEY_PLANAR_Defend:
            return 0

        # traj_ee_pos = np.vstack(base_env.traj_ee_pos)
        # traj_ee_vel = np.vstack(base_env.traj_ee_vel)
        traj_puck_pos = np.vstack(base_env.traj_puck_pos)
        traj_puck_vel = np.vstack(base_env.traj_puck_vel)
        # traj_cos_ang = np.stack(base_env.traj_cos_ang)

        # get score
        if base_env.has_goal and base_env.has_hit:
            return -20

        if base_env.has_goal:
            return -10

        if base_env.has_hit:
            puck_pos = traj_puck_pos[base_env.hit_step - 1, 0:2]
            puck_vel = traj_puck_vel[base_env.hit_step - 1, 0:2]

            top_y_pos = env_info["table"]["width"] / 2

            goal_x_pos = -env_info["table"]["length"] / 2
            goal_y_r = env_info["table"]["goal_width"] / 2

            goal_top_pos = [goal_x_pos, goal_y_r]
            goal_bot_pos = [goal_x_pos, -goal_y_r]

            theta_top = np.degrees(angle_2d(goal_top_pos - puck_pos, puck_vel))
            theta_bot = np.degrees(angle_2d(goal_bot_pos - puck_pos, puck_vel))

            towards = (theta_top * theta_bot < 0) and (np.abs(theta_bot) < 90) and (np.abs(theta_top) < 90)

            if towards:
                danger1 = (np.abs(theta_bot) + np.abs(theta_top)) / 180

                max_dist = np.linalg.norm(np.array([0, top_y_pos]) - np.array([goal_x_pos, 0]))
                puck_goal_dist = np.linalg.norm(np.array(puck_pos) - np.array([goal_x_pos, 0]))
                danger2 = 1 - max(puck_goal_dist / max_dist, 1)

                danger = max(danger1, danger2)

                return 15 + 5 * (1 - danger)

        return 0

    @staticmethod
    def planar_defend_sparse_SuperSimple(base_env: AirHockeyDefend, state, action, next_state, absorbing):
        env_info = base_env.env_info

        if base_env.ep_step < MAX_EPISODE_STEPS_AIR_HOCKEY_PLANAR_Defend:
            return 0

        # get score
        # -0.8 < puck_pos[0] <= -0.29 and puck_vel[0] < 0.1
        if base_env.has_success:
            return 20

        return -20

    @staticmethod
    def planar_defend_sparse_SuperSimpleAdditive(base_env: AirHockeyDefend, state, action, next_state, absorbing):
        env_info = base_env.env_info

        # no need to collect trajectories as we only reward based on the end state of the puck

        if base_env.ep_step < MAX_EPISODE_STEPS_AIR_HOCKEY_PLANAR_Defend:
            return 0

        r = 0
        puck_pos, puck_vel = base_env.get_puck(next_state)

        # -0.8 < puck_pos[0] <= -0.29 and puck_vel[0] < 0.1
        r += 6 if -0.8 < puck_pos[0] else -6
        r += 10 if puck_pos[0] <= -0.29 and (not base_env.has_goal) else -10
        r += 4 if puck_vel[0] < 0.1 else -4

        return r

    @staticmethod
    def planar_defend_sparse_FavorSuccessRegion(base_env: AirHockeyDefend, state, action, next_state, absorbing):
        env_info = base_env.env_info

        if base_env.episode_steps < MAX_EPISODE_STEPS_AIR_HOCKEY_PLANAR_Defend:
            return 0

        traj_ee_pos = np.vstack(base_env.ee_pos_history)
        traj_ee_vel = np.vstack(base_env.ee_vel_history)
        traj_puck_pos = np.vstack(base_env.puck_pos_history)
        traj_puck_vel = np.vstack(base_env.puck_vel_history)

        rim_negativity = 1
        if base_env.has_bounce_rim_home:
            table_width = env_info["table"]["width"]
            goal_width = env_info["table"]["goal_width"]
            bounce_len = (table_width / 2) - (goal_width / 2)
            puck_y = traj_puck_pos[base_env.has_bounce_rim_home_step, 1]
            puck_len = np.abs(puck_y) - (goal_width / 2)

            rim_negativity = 1 - (puck_len / bounce_len)

        reward = 0

        reward += (1/(1+base_env.last_puck_dist_to_success_region_x)) * 20

        # always reward success
        if base_env.has_success:
            reward += 50

        if base_env.has_goal:
            reward += -20
        elif base_env.has_bounce_rim_home:
            reward += rim_negativity * (-10)

        # take success region oscillation into account
        if base_env.success_region_change_count % 2 == 0:
            # not in succ region
            reward += np.max([-20, -7.5 * base_env.success_region_change_count])
        else:
            # in succ region
            reward += np.max([5, 35 + (-10 * base_env.success_region_change_count)])

        if base_env.last_puck_vel_x < 0.1:
            reward += 10

        if base_env.has_hit:
            reward += 5

        return reward

    @staticmethod
    def planar_defend_DongxuV5(base_env: AirHockeyDefend, obs, action, obs_, absorbing):

        r = 0
        puck_pos, puck_vel = base_env.get_puck(obs_)
        ee_pos, _ = base_env.get_ee()
        env_info = base_env.env_info
        is_missed = puck_pos[0] < ee_pos[0]

        # If absorbing the puck is out of bounds of the table.
        if absorbing:
            # large penalty if agent coincides a goal
            if puck_pos[0] + env_info['table']['length'] / 2 < 0 and \
                    np.abs(puck_pos[1]) - env_info['table']['goal_width'] < 0:
                r = -50
        else:
            # If the puck bounced off the head walls, there is no reward.
            if base_env.has_bounce_rim_home:
            # if is_missed:
                r = -1
            elif base_env.has_hit:
                # Reward if the puck slows down on the defending side
                r = 5
                if -0.8 < puck_pos[0] < -0.4:
                    r_y = 3 * np.exp(-3 * np.abs(puck_pos[1]))
                    r_x = np.exp(-5 * np.abs(puck_pos[0] + 0.6))
                    r_vel = 5 * np.exp(-(5 * np.linalg.norm(puck_vel)) ** 2)
                    r += r_x + r_y + r_vel

                # If we did not yet hit the puck, reward is controlled by the distance between end effector and puck
                # on the x axis
            else:
                # Maybe change -0.6 to -0.4 so the puck is stopped a bit higher, could improve performance because
                # we don't run into the constraints at the bottom
                ee_des = np.array([puck_pos[0], puck_pos[1]])
                dist_ee_puck = np.abs(ee_des - ee_pos[:2])

                r_x = np.exp(-3 * dist_ee_puck[0])

                sig = 0.4
                r_y = 1.5 / (np.sqrt(2. * np.pi) * sig) * np.exp(-np.power((dist_ee_puck[1] - 0.08) / sig, 2.) / 2)

                r = 0.3 * r_x + 0.7 * r_y

        # penalty for violation of constraints
        constraints_value = copy.deepcopy(env_info['constraints'].fun(obs[env_info['joint_pos_ids']],
                                                                 obs[env_info['joint_vel_ids']]))

        constr_dict = {'joint_pos_constr': [-2.9, +2.9, -1.8, +1.8, -2.0, +2.0],
                       'joint_vel_constr': [-1.5, +1.5, -1.5, +1.5, -2.0, +2.0],
                       'ee_constr': [-env_info['table']['length']/2, -env_info['table']['width']/2, env_info['table']['width']/2, -0.02, 0.02]}

        r_constraint = 0
        for key, constr in constraints_value.items():
            for i, element in enumerate(constr):
                if element > 0:
                    ratio = element / abs(constr_dict[key][i])
                    r_constraint += np.exp(-ratio)

        jerk = base_env.jerk
        for element in jerk:
            if element > 0:
                ratio = element / 10000
                r_constraint += -0.01 * ratio

        r += r_constraint

        # penalizes the amount of torque used
        r -= 0.001 * np.linalg.norm(action)
        return r

reward_functions = {
    'DongxuV5': AirHockeyPlanarDefend.planar_defend_DongxuV5,
    'FavorSuccessRegion': AirHockeyPlanarDefend.planar_defend_sparse_FavorSuccessRegion,
}

def angle_2d(v0, v1):
    return np.math.atan2(np.linalg.det([v1,v0]),np.dot(v0,v1))

def mp(text):
    def inner(x):
        print(text, x)
        return x
    return inner