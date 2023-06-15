import numpy as np
from typing import Union, Tuple

from .air_hockey_hit import AirHockeyGymHit
from .air_hockey_utils import TrajectoryOptimizer
from scipy.interpolate import CubicSpline


class AirHockeyGymHitCart(AirHockeyGymHit):
    def __init__(self, env_id=None,
                 interpolation_order=3, custom_reward_function='HitSparseRewardV0',
                 check_step=True, check_traj=True, check_traj_length=-1):
        super().__init__(env_id=env_id,
                         interpolation_order=interpolation_order,
                         custom_reward_function=custom_reward_function,
                         check_step=check_step,
                         check_traj=check_traj,
                         check_traj_length=check_traj_length)

        self.traj_opt = TrajectoryOptimizer(self.env_info)

        self.sub_traj_idx = 0
        if self.dof == 3:
            self.q_prev = np.array([-1.15570, +1.30024, +1.44280])
            self.dq_prev = np.zeros([3])
            self.ddq_prev = np.zeros([3])
        else:
            self.q_prev = np.array([0., -0.1961, 0., -1.8436, 0., +0.9704, 0.])
            self.dq_prev = np.zeros([7])
            self.ddq_prev = np.zeros(([7]))

        # constr
        if self.dof == 3:
            self.constr_ee = np.array([[+0.585, +1.585], [-0.470, +0.470], [+0.080, +0.120]])
        else:
            self.constr_ee = np.array([[+0.585, +1.685], [-0.470, +0.470], [+0.1245, +0.2045]])

    def reset(self, **kwargs):

        self.sub_traj_idx = 0
        if self.dof == 3:
            self.q_prev = np.array([-1.15570, +1.30024, +1.44280])
            self.dq_prev = np.zeros([3])
            self.ddq_prev = np.zeros([3])
        else:
            self.q_prev = np.array([0., -0.1961, 0., -1.8436, 0., +0.9704, 0.])
            self.dq_prev = np.zeros([7])
            self.ddq_prev = np.zeros(([7]))

        return super().reset(**kwargs)

    def check_traj_validity(self, action, traj_pos, traj_vel):
        sub_traj_length = self.check_traj_length[self.sub_traj_idx]
        if sub_traj_length != -1:
            valid_pos = traj_pos[:sub_traj_length]
            valid_vel = traj_vel[:sub_traj_length]
        else:
            valid_pos = traj_pos
            valid_vel = traj_vel
        self.sub_traj_idx += 1

        if self.check_traj:

            # check tau
            invalid_tau = False
            # if action.shape[0] % 2 != 0:
            #     tau_bound = [1.5, 3.0]
            #     invalid_tau = action[0] < tau_bound[0] or action[0] > tau_bound[1]

            # check ee constr
            constr_ee = self.constr_ee
            invalid_ee = np.any(valid_pos < constr_ee[:, 0]) or np.any(valid_pos > constr_ee[:, 1])

            if invalid_tau or invalid_ee:
                # traj_pos = np.stack([self.q_prev] * valid_pos.shape[0], axis=0)
                # traj_vel = np.stack([self.dq_prev] * valid_vel.shape[0], axis=0)
                return False, valid_pos, valid_vel

        # optimize traj
        q_start = self.q_prev
        dq_start = self.dq_prev
        ddq_start = self.ddq_prev
        success, traj_q_opt = self.traj_opt.optimize_trajectory(np.hstack([valid_pos, valid_vel]),
                                                                q_start, dq_start, None)
        t = np.linspace(0, traj_q_opt.shape[0], traj_q_opt.shape[0] + 1) * 0.02
        f = CubicSpline(t, np.vstack([q_start, traj_q_opt]), axis=0, bc_type=((1, dq_start), (2, ddq_start)))
        df = f.derivative(1)
        ddf = f.derivative(2)
        traj_pos = np.array(f(t[1:]))
        traj_vel = np.array(df(t[1:]))
        self.q_prev = np.array(f(t[-1]))
        self.dq_prev = np.array(df(t[-1]))
        self.ddq_prev = np.array(ddf(t[-1]))
        return True, traj_pos, traj_vel

    def get_invalid_traj_penalty(self, action, traj_pos, traj_vel):
        # violate tau penalty
        violate_tau_penalty = 0
        # if action.shape[0] % 2 != 0:
        #     tau_bound = [1.5, 3.0]
        #     violate_tau_penalty = np.max([0, action[0] - tau_bound[1]]) + np.max([0, tau_bound[0] - action[0]])

        sub_traj_length = self.check_traj_length[self.sub_traj_idx-1]
        if sub_traj_length != -1:
            valid_pos = traj_pos[:sub_traj_length]
            valid_vel = traj_vel[:sub_traj_length]
        else:
            valid_pos = traj_pos
            valid_vel = traj_vel

        # violate ee penalty
        constr_ee = self.constr_ee
        num_violate_ee_constr = np.array((valid_pos - constr_ee[:, 0] < 0), dtype=np.float32).mean() + \
                                np.array((valid_pos - constr_ee[:, 1] > 0), dtype=np.float32).mean()
        max_violate_ee_constr = np.maximum(constr_ee[:, 0] - valid_pos, 0).mean() + \
                                np.maximum(valid_pos - constr_ee[:, 1], 0).mean()
        violate_ee_penalty = num_violate_ee_constr + max_violate_ee_constr

        traj_invalid_penalty = violate_tau_penalty + violate_ee_penalty
        return -3 * traj_invalid_penalty

    def get_invalid_traj_return(self, action, traj_pos, traj_vel):
        obs, rew, done, info = self.step(np.hstack([self.q_prev, self.dq_prev]))

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


class AirHockey3DofHitCart(AirHockeyGymHitCart):
    def __init__(self, interpolation_order=3, custom_reward_function='HitSparseRewardV0',
                 check_step=True, check_traj=True, check_traj_length=-1):
        super().__init__(env_id="3dof-hit",
                         interpolation_order=interpolation_order,
                         custom_reward_function=custom_reward_function,
                         check_step=check_step,
                         check_traj=check_traj,
                         check_traj_length=check_traj_length)


class AirHockey7DofHitCart(AirHockeyGymHitCart):
    def __init__(self, interpolation_order=3, custom_reward_function='HitSparseRewardV0',
                 check_step=True, check_traj=True, check_traj_length=-1):
        super().__init__(env_id="7dof-hit",
                         interpolation_order=interpolation_order,
                         custom_reward_function=custom_reward_function,
                         check_step=check_step,
                         check_traj=check_traj,
                         check_traj_length=check_traj_length)