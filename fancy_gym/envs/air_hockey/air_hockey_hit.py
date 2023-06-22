from typing import Union, Tuple, Optional
import copy
import numpy as np
from gym import spaces, utils
from gym.core import ObsType, ActType
from fancy_gym.envs.air_hockey.air_hockey import AirHockeyGymBase

# from air_hockey_challenge.utils import robot_to_world
# from air_hockey_challenge.framework import AirHockeyChallengeWrapper
# from air_hockey_challenge.environments.planar import AirHockeyHit, AirHockeyDefend
from air_hockey_challenge.utils import forward_kinematics, robot_to_world

MAX_EPISODE_STEPS_AIR_HOCKEY_3DOF_HIT = 150  # default is 500, recommended 120
MAX_EPISODE_STEPS_AIR_HOCKEY_7DOF_HIT = 150  # default is 500, recommended 120


class AirHockeyGymHit(AirHockeyGymBase):
    def __init__(self, env_id=None,
                 interpolation_order=None,
                 custom_reward_function=None,
                 check_step=True,
                 check_traj=True,
                 check_traj_length=-1,
                 early_stop=False,
                 wait_puck=False):

        custom_reward_functions = {
            'HitRewardDefault': self.hit_reward_default,
            'HitSparseRewardV0': self.hit_sparse_reward_v0,
            'HitSparseRewardV1': self.hit_sparse_reward_v1,
            'HitSparseRewardV2': self.hit_sparse_reward_v2,
            'HitSparseRewardWaitPuck': lambda *args: self.hit_sparse_reward_wait_puck(*args),
            'HitSparseRewardEnes': lambda *args: self.hit_reward_enes(*args)
        }
        super().__init__(env_id=env_id,
                         interpolation_order=interpolation_order,
                         custom_reward_function=custom_reward_functions[custom_reward_function])

        # modify observation space
        if '3dof' in env_id:
            obs_dim = 12
        else:
            obs_dim = 23
        obs_l = np.ones(obs_dim) * -10000
        obs_h = np.ones(obs_dim) * +10000
        self.observation_space = spaces.Box(low=obs_l, high=obs_h, dtype=np.float32)

        # mp wrapper related term
        if interpolation_order is None:
            self.dt = 0.001
        else:
            self.dt = 0.001
        if '3dof' in env_id:
            self.horizon = MAX_EPISODE_STEPS_AIR_HOCKEY_3DOF_HIT
        else:
            self.horizon = MAX_EPISODE_STEPS_AIR_HOCKEY_7DOF_HIT

        # constraints
        if '3dof' in env_id:
            self.constr_j_pos = np.array([[-2.81, +2.81], [-1.70, +1.70], [-1.98, +1.98]])
            self.constr_j_vel = np.array([[-1.49, +1.49], [-1.49, +1.49], [-1.98, +1.98]])
            self.constr_ee = np.array([[+0.585, +1.585], [-0.470, +0.470], [+0.080, +0.120]])
        else:
            self.constr_j_pos = np.array([[-2.967, +2.967], [-2.094, +2.094], [-2.967, +2.967], [-2.094, +2.094],
                                          [-2.967, +2.967], [-2.094, +2.094], [-3.053, +3.054]])
            self.constr_j_vel = np.array([[-1.483, +1.483], [-1.483, +1.483], [-1.745, +1.745], [-1.308, +1.308],
                                          [-2.268, +2.268], [-2.356, +2.356], [-2.356, +2.356]])
            self.constr_ee = np.array([[+0.585, +1.585], [-0.470, +0.470], [+0.1245, +0.2045]])

        self.sub_traj_idx = 0
        if self.dof == 3:
            self.q_prev = np.array([-1.15570, +1.30024, +1.44280])
            self.dq_prev = np.zeros([3])
            self.ddq_prev = np.zeros([3])
        else:
            self.q_prev = np.array([0., -0.1961, 0., -1.8436, 0., +0.9704, 0.])
            self.dq_prev = np.zeros([7])
            self.ddq_prev = np.zeros(([7]))

        # reward related terms
        self.positive_reward_coef = 1
        self.received_hit_rew = False
        self.received_sparse_rew = False

        # validity checking
        self.check_step = check_step
        self.check_traj = check_traj
        self.step_penalty_coef = 0.05
        self.traj_penalty_coef = 1
        self.check_traj_length = check_traj_length

        self.early_stop = early_stop
        self.wait_puck = wait_puck
        self.wait_puck_steps = 0

    def reset(self, **kwargs):
        self.received_hit_rew = False
        self.received_sparse_rew = False

        self.sub_traj_idx = 0
        if self.dof == 3:
            self.q_prev = np.array([-1.15570, +1.30024, +1.44280])
            self.dq_prev = np.zeros([3])
            self.ddq_prev = np.zeros([3])
        else:
            self.q_prev = np.array([0., -0.1961, 0., -1.8436, 0., +0.9704, 0.])
            self.dq_prev = np.zeros([7])
            self.ddq_prev = np.zeros(([7]))

        obs = super().reset(**kwargs)

        self.wait_puck_steps = 0
        if self.wait_puck:
            p_v_x, p_v_y, p_v_a = obs[3], obs[4], np.abs(obs[5])
            p_v_l = np.sqrt(p_v_x**2 + p_v_y**2)
            while p_v_l > 0.05 and p_v_a > 0.05:
                self.wait_puck_steps += 1
                obs, _, _, _ = self.env.step(np.vstack([self.q_prev, self.dq_prev]))
                # self.env.render()
                p_v_x, p_v_y, p_v_a = obs[3], obs[4], np.abs(obs[5])
                p_v_l = np.sqrt(p_v_x ** 2 + p_v_y ** 2)
        # print(self.wait_puck_steps)

        return np.array(obs, dtype=np.float32)

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        obs, rew, done, info = super().step(action)

        # check step validity
        step_validity, step_penalty = self.check_step_validity(info)
        if not step_validity and self.early_stop:
            return obs, step_penalty, True, info

        rew = rew if step_validity else rew + step_penalty
        return obs, rew, done, info

    def check_step_validity(self, info):
        info["has_hit"] = 1 if info["has_hit"] else 0
        info["has_hit_step"] = self.horizon if info["has_hit_step"] == 500 else info["has_hit_step"]
        info["has_scored"] = 1 if info["has_scored"] else 0
        info["has_scored_step"] = self.horizon if info["has_scored_step"] == 500 else info["has_scored_step"]

        # info["validity"] = 1 if validity else 0
        info["ee_violation"] = np.any(info['constraints_value']['ee_constr'] > 0).astype(int)
        info["jerk_violation"] = np.any(info['jerk'][:self.dof] > 1e4).astype(int)
        info["j_pos_violation"] = np.any(info['constraints_value']['joint_pos_constr'] > 0).astype(int)
        info["j_vel_violation"] = np.any(info['constraints_value']['joint_vel_constr'] > 0).astype(int)

        validity, penalty = True, 0
        if self.check_step:
            coef = (self.horizon - self._episode_steps) / self.horizon
            if self.early_stop:
                ee_constr = np.array(np.any(info['constraints_value']['ee_constr'] > 0), dtype=np.float32)
                # jerk_constr = np.array((info['jerk'] > 1e4), dtype=np.float32).mean()
                j_pos_constr = np.array((info['constraints_value']['joint_pos_constr'] > 0), dtype=np.float32).mean()
                j_vel_constr = np.array((info['constraints_value']['joint_vel_constr'] > 0), dtype=np.float32).mean()
                penalty = coef * (ee_constr + np.tanh(j_pos_constr) + np.tanh(j_vel_constr))
            else:
                ee_constr = np.maximum(info['constraints_value']['ee_constr'], 0).mean()
                # jerk_constr = np.maximum(info['jerk'] - 1e4, 0).mean()
                j_pos_constr = np.maximum(info['constraints_value']['joint_pos_constr'], 0).mean()
                j_vel_constr = np.maximum(info['constraints_value']['joint_vel_constr'], 0).mean()
                penalty = coef * (np.tanh(ee_constr) + np.tanh(j_pos_constr) + np.tanh(j_vel_constr)) / 20
            validity = False if penalty > 0 else True
            penalty = -penalty if penalty > 0 else 0

        return validity, penalty

    def get_invalid_step_penalty(self, info):
        pass

    def get_invalid_step_return(self, info):
        pass

    def check_traj_validity(self, action, traj_pos, traj_vel):
        if not self.check_traj:
            return True, traj_pos, traj_vel

        # check tau
        invalid_tau = False
        # if action.shape[0] % 3 != 0:
        #     tau_bound = [1.5, 3.0]
        #     invalid_tau = action[0] < tau_bound[0] or action[0] > tau_bound[1]

        if self.check_traj_length != -1:
            valid_pos = traj_pos[:self.check_traj_length]
            valid_vel = traj_vel[:self.check_traj_length]
        else:
            valid_pos = traj_pos
            valid_vel = traj_vel

        # check joint constr
        constr_j_pos = self.constr_j_pos
        constr_j_vel = self.constr_j_vel
        invalid_j_pos = np.any(valid_pos < constr_j_pos[:, 0]) or np.any(valid_pos > constr_j_pos[:, 1])
        invalid_j_vel = np.any(valid_vel < constr_j_vel[:, 0]) or np.any(valid_vel > constr_j_vel[:, 1])

        if invalid_tau or invalid_j_pos or invalid_j_vel:
            return False, traj_pos, traj_vel
        return True, traj_pos, traj_vel

    def get_invalid_traj_penalty(self, action, traj_pos, traj_vel):
        # violate tau penalty
        violate_tau_penalty = 0
        # if action.shape[0] % 3 != 0:
        #     tau_bound = [1.5, 3.0]
        #     violate_tau_penalty = np.max([0, action[0] - tau_bound[1]]) + np.max([0, tau_bound[0] - action[0]])

        if self.check_traj_length != -1:
            valid_pos = traj_pos[:self.check_traj_length]
            valid_vel = traj_vel[:self.check_traj_length]
        else:
            valid_pos = traj_pos
            valid_vel = traj_vel

        # violate joint penalty
        constr_j_pos = self.constr_j_pos
        constr_j_vel = self.constr_j_vel
        num_violate_j_pos_constr = np.array((valid_pos - constr_j_pos[:, 0] < 0), dtype=np.float32).mean() + \
                                   np.array((valid_pos - constr_j_pos[:, 1] > 0), dtype=np.float32).mean()
        num_violate_j_vel_constr = np.array((valid_vel - constr_j_vel[:, 0] < 0), dtype=np.float32).mean() + \
                                   np.array((valid_vel - constr_j_vel[:, 1] > 0), dtype=np.float32).mean()
        max_violate_j_pos_constr = np.maximum(constr_j_pos[:, 0] - valid_pos, 0).mean() + \
                                   np.maximum(valid_pos - constr_j_pos[:, 1], 0).mean()
        max_violate_j_vel_constr = np.maximum(constr_j_vel[:, 0] - valid_vel, 0).mean() + \
                                   np.maximum(valid_vel - constr_j_vel[:, 1], 0).mean()
        violate_j_pos_penalty = num_violate_j_pos_constr + max_violate_j_pos_constr
        violate_j_vel_penalty = num_violate_j_vel_constr + max_violate_j_vel_constr

        traj_invalid_penalty = violate_tau_penalty + violate_j_pos_penalty + violate_j_vel_penalty

        if self.interpolation_order is None:
            coef = 20 * self.dof
        else:
            coef = self.dof
        return -coef * traj_invalid_penalty

    def get_invalid_traj_return(self, action, traj_pos, traj_vel):
        obs, rew, done, info = self.step(np.hstack([traj_pos[0], traj_vel[0]]))

        # in fancy gym added metrics
        info["validity"] = 0
        info["ee_violation"] = 1
        info["jerk_violation"] = 1
        info["j_pos_violation"] = 1
        info["j_vel_violation"] = 1

        # default metrics
        info["has_hit"] = 0
        info["has_hit_step"] = self.horizon
        info["has_scored"] = 0
        info["has_scored_step"] = self.horizon
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
            info[k] = [v] * 25

        info['trajectory_length'] = 1

        return obs, self.get_invalid_traj_penalty(action, traj_pos, traj_vel), True, info

    @staticmethod
    def hit_reward_default(base_env, obs, act, obs_, done):
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
    def hit_sparse_reward_v0(base_env, obs, act, obs_, done):
        # init reward and env info
        env_info = base_env.env_info
        goal_pos = np.array([env_info["table"]["length"] / 2, 0, 0])

        if base_env.episode_steps < MAX_EPISODE_STEPS_AIR_HOCKEY_3DOF_HIT and not done:
            return 0.01

        traj_ee_pos = np.vstack(base_env.ee_pos_history)
        traj_ee_vel = np.vstack(base_env.ee_vel_history)
        traj_puck_pos = np.vstack(base_env.puck_pos_history)
        traj_puck_vel = np.vstack(base_env.puck_vel_history)
        traj_cos_ee_puck_goal = np.stack(base_env.cos_ee_puck_goal_history)

        # get score
        if base_env.has_scored:
            coef = np.clip(1.0 - (100 - base_env.has_scored_step) / 50, 0, 1)
            success_reward = 16
            return 4 + coef * success_reward

        if base_env.has_hit:
            has_hit_step = base_env.has_hit_step
            cos_ee_puck_goal = traj_cos_ee_puck_goal[has_hit_step]
            min_dist_puck_goal = base_env.min_dist_puck_goal
            max_p_vel_x = np.max(traj_puck_vel[:, 0])
            return 1 + 1.0 * cos_ee_puck_goal + 1.0 * (1 - np.tanh(min_dist_puck_goal)) + 1.0 * np.tanh(max_p_vel_x)

        idx = np.argmin(np.linalg.norm(traj_puck_pos - traj_ee_pos, axis=1))
        coef = traj_cos_ee_puck_goal[idx]
        min_dist_ee_puck = np.linalg.norm(traj_puck_pos[idx] - traj_ee_pos[idx])
        return coef * (1 - np.tanh(min_dist_ee_puck))  # [0, 1]

    @staticmethod
    def hit_sparse_reward_v1(base_env, obs, act, obs_, done):
        # init reward and env info
        env_info = base_env.env_info
        goal_pos = np.array([env_info["table"]["length"] / 2, 0, 0])

        if base_env.episode_steps < MAX_EPISODE_STEPS_AIR_HOCKEY_3DOF_HIT and not done:
            return 0.01

        traj_ee_pos = np.vstack(base_env.ee_pos_history)
        traj_ee_vel = np.vstack(base_env.ee_vel_history)
        traj_puck_pos = np.vstack(base_env.puck_pos_history)
        traj_puck_vel = np.vstack(base_env.puck_vel_history)
        traj_cos_ee_puck_goal = np.stack(base_env.cos_ee_puck_goal_history)

        # get score
        if base_env.has_scored:
            coef = np.clip(1.0 - (base_env.has_scored_step - 50) / 100, 0, 1)
            success_reward = 16
            return 4 + coef * success_reward  # [4, 10]

        if base_env.has_hit:
            has_hit_step = base_env.has_hit_step
            min_dist_puck_goal = base_env.min_dist_puck_goal
            max_puck_vel_after_hit = base_env.max_puck_vel_after_hit
            mean_puck_vel_after_hit = base_env.mean_puck_vel_after_hit
            return 1 + 1.0 * (1 - np.tanh(min_dist_puck_goal)) + \
                1.0 * np.tanh(max_puck_vel_after_hit) + 1.0 * np.tanh(mean_puck_vel_after_hit)  # [1, 4]

        idx = np.argmin(np.linalg.norm(traj_puck_pos - traj_ee_pos, axis=1))
        coef = traj_cos_ee_puck_goal[idx]
        min_dist_ee_puck = np.linalg.norm(traj_puck_pos[idx] - traj_ee_pos[idx])
        return 1 - np.tanh(min_dist_ee_puck)  # [0, 1]

    @staticmethod
    def hit_sparse_reward_v2(base_env, obs, act, obs_, done):
        # sparse reward
        if base_env.episode_steps < MAX_EPISODE_STEPS_AIR_HOCKEY_3DOF_HIT and not done:
            if base_env.episode_steps < 100:
                return 0.01
            else:
                return 0

        # init env info
        env_info = base_env.env_info
        goal_pos = np.array([env_info["table"]["length"] / 2, 0, 0])

        traj_ee_pos = np.vstack(base_env.ee_pos_history)
        traj_ee_vel = np.vstack(base_env.ee_vel_history)
        traj_puck_pos = np.vstack(base_env.puck_pos_history)
        traj_puck_vel = np.vstack(base_env.puck_vel_history)
        traj_cos_ee_puck_goal = np.stack(base_env.cos_ee_puck_goal_history)
        traj_cos_ee_puck_bouncing_point = np.stack(base_env.cos_ee_puck_bouncing_point_history)

        # get score
        if base_env.has_scored:
            coef = np.clip(1.0 - (base_env.has_scored_step - 80) / 100, 0, 1)
            coef_rew = 6
            success_rew = 8
            max_puck_vel_after_hit = base_env.max_puck_vel_after_hit
            mean_puck_vel_after_hit = base_env.mean_puck_vel_after_hit
            return 4 + 1.0 * np.tanh(max_puck_vel_after_hit) + 1.0 * np.tanh(mean_puck_vel_after_hit) + \
                coef * coef_rew + 1.0 * success_rew  # [4, 20]

        if base_env.has_hit:
            has_hit_step = base_env.has_hit_step
            cos_ee_puck_goal = traj_cos_ee_puck_goal[has_hit_step]
            cos_ee_puck_bouncing_point = traj_cos_ee_puck_bouncing_point[has_hit_step]
            # cos_rew = np.max([cos_ee_puck_goal, cos_ee_puck_bouncing_point])
            cos_rew = cos_ee_puck_goal
            min_dist_puck_goal = base_env.min_dist_puck_goal
            return 1 * (1 + 1.0 * cos_rew + 2.0 * (1 - np.tanh(min_dist_puck_goal)))  # [1, 4]

        min_dist_ee_puck = np.min(np.linalg.norm(traj_puck_pos - traj_ee_pos, axis=1))
        return 1 * (1 - np.tanh(min_dist_ee_puck))  # [0, 1]

    def hit_sparse_reward_wait_puck(self, *args):
        return self._hit_sparse_reward_wait_puck(*args)

    def _hit_sparse_reward_wait_puck(self, base_env, obs, act, obs_, done):
        # sparse reward
        if base_env.episode_steps - self.wait_puck_steps < MAX_EPISODE_STEPS_AIR_HOCKEY_3DOF_HIT and not done:
            if base_env.episode_steps - self.wait_puck_steps < 100:
                return 0.01
            else:
                return 0

        # init env info
        env_info = base_env.env_info
        goal_pos = np.array([env_info["table"]["length"] / 2, 0, 0])

        if self.wait_puck_steps >= len(base_env.ee_pos_history):
            return 0

        traj_ee_pos = np.vstack(base_env.ee_pos_history[self.wait_puck_steps:])
        traj_ee_vel = np.vstack(base_env.ee_vel_history[self.wait_puck_steps:])
        traj_puck_pos = np.vstack(base_env.puck_pos_history[self.wait_puck_steps:])
        traj_puck_vel = np.vstack(base_env.puck_vel_history[self.wait_puck_steps:])
        traj_cos_ee_puck_goal = np.stack(base_env.cos_ee_puck_goal_history[self.wait_puck_steps:])
        traj_cos_ee_puck_bouncing_point = np.stack(base_env.cos_ee_puck_bouncing_point_history[self.wait_puck_steps:])

        # get score
        if base_env.has_scored:
            if self.wait_puck_steps >= base_env.has_scored_step:
                return 0
            has_scored_step = base_env.has_scored_step - self.wait_puck_steps
            coef = np.clip(1.0 - (has_scored_step - 80) / 100, 0, 1)
            coef_rew = 6
            success_rew = 8
            max_puck_vel_after_hit = base_env.max_puck_vel_after_hit
            mean_puck_vel_after_hit = base_env.mean_puck_vel_after_hit
            return 4 + 1.0 * np.tanh(max_puck_vel_after_hit) + 1.0 * np.tanh(mean_puck_vel_after_hit) + \
                coef * coef_rew + 1.0 * success_rew  # [4, 20]

        if base_env.has_hit:
            if self.wait_puck_steps >= base_env.has_hit_step:
                return 0
            has_hit_step = base_env.has_hit_step - self.wait_puck_steps
            cos_ee_puck_goal = traj_cos_ee_puck_goal[has_hit_step]
            cos_ee_puck_bouncing_point = traj_cos_ee_puck_bouncing_point[has_hit_step]
            # cos_rew = np.max([cos_ee_puck_goal, cos_ee_puck_bouncing_point])
            cos_rew = cos_ee_puck_goal
            min_dist_puck_goal = base_env.min_dist_puck_goal
            return 1 * (1 + 1.0 * cos_rew + 2.0 * (1 - np.tanh(min_dist_puck_goal)))  # [1, 4]

        min_dist_ee_puck = np.min(np.linalg.norm(traj_puck_pos - traj_ee_pos, axis=1))
        return 1 * (1 - np.tanh(min_dist_ee_puck))  # [0, 1]

    def hit_reward_enes(self, *args):
        return self._hit_reward_enes(*args)

    def _hit_reward_enes(self, base_env, obs, act, obs_, done):
        env_info = base_env.env_info
        positive_reward_coef = self.positive_reward_coef
        rew = 0.0001 * positive_reward_coef

        ee_pos, ee_vel = base_env.get_ee()  # current ee state
        puck_pos, puck_vel = base_env.get_puck(obs_)  # current puck state
        if not base_env.has_hit:
            old_ee_pos_in_robot, _ = forward_kinematics(env_info["robot"]["robot_model"],
                                                        env_info["robot"]["robot_data"],
                                                        base_env.get_joints(obs)[0])
            old_ee_pos_in_world, _ = robot_to_world(env_info["robot"]["base_frame"][0], old_ee_pos_in_robot)
            ee_puck_dis = np.linalg.norm(ee_pos[:2] - puck_pos[:2])
            old_ee_puck_dis = np.linalg.norm(old_ee_pos_in_world[:2] - puck_pos[:2])
            rew += 0.01 * np.exp(-5 * ee_puck_dis) * (old_ee_puck_dis - ee_puck_dis > 0) * positive_reward_coef
        else:
            rew += 0.01 * positive_reward_coef

        if base_env.has_hit and not self.received_hit_rew:
            self.received_hit_rew = True
            rew += (1 + puck_vel[0] ** 2) * positive_reward_coef

        if self.received_sparse_rew:
            return rew

        vel_comp = np.tanh(1 / 2 * np.abs(puck_vel[0]))
        if base_env.has_scored:
            rew += (100 * vel_comp) * positive_reward_coef
            rew += 100 * positive_reward_coef
            self.received_sparse_rew = True
        elif base_env.has_bounce_rim_away:
            # Can still score so wait for the next steps
            if puck_vel[0] > 0:
                return rew
            # Bounced back from opponent
            else:
                env_info = base_env.env_info
                goal_width = env_info["table"]["goal_width"] / 2
                dist_comp = 1 - np.tanh(3 * (np.abs(puck_pos[1]) - goal_width))
                rew += (10 + 5 * dist_comp + 5 * vel_comp) * positive_reward_coef
                self.received_sparse_rew = True
        # We hit the puck backwards
        elif not base_env.has_bounce_rim_away and puck_vel[0] < 0:
            rew += -1 * positive_reward_coef
            self.received_sparse_rew = True

        return rew


class AirHockey3DofHit(AirHockeyGymHit):
    def __init__(self, interpolation_order=3, custom_reward_function='HitSparseRewardV0',
                 check_step=True, check_traj=True, check_traj_length=-1):

        super().__init__(env_id="3dof-hit",
                         interpolation_order=interpolation_order,
                         custom_reward_function=custom_reward_function,
                         check_step=check_step,
                         check_traj=check_traj,
                         check_traj_length=check_traj_length)


class AirHockey7DofHit(AirHockeyGymHit):
    def __init__(self, interpolation_order=3, custom_reward_function='HitSparseRewardV0',
                 check_step=True, check_traj=True, check_traj_length=-1):

        super().__init__(env_id="7dof-hit",
                         interpolation_order=interpolation_order,
                         custom_reward_function=custom_reward_function,
                         check_step=check_step,
                         check_traj=check_traj,
                         check_traj_length=check_traj_length)
