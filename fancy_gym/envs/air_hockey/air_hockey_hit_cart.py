import numpy as np
from typing import Union, Tuple
from gym import spaces, utils
from gym.core import ObsType, ActType

from .air_hockey_hit import AirHockeyGymHit
from .air_hockey_utils import TrajectoryOptimizer
from scipy.interpolate import CubicSpline


class AirHockeyGymHitCart(AirHockeyGymHit):
    def __init__(self, env_id=None,
                 interpolation_order=3, custom_reward_function='HitSparseRewardV0',
                 check_step=True, check_step_stop=False,
                 check_traj=True, check_traj_length=-1,
                 add_obs_noise: bool = False, use_obs_estimator: bool = False,
                 wait_puck=False):
        super().__init__(env_id=env_id,
                         interpolation_order=interpolation_order,
                         custom_reward_function=custom_reward_function,
                         check_step=check_step,
                         check_step_stop=check_step_stop,
                         check_traj=check_traj,
                         check_traj_length=check_traj_length,
                         add_obs_noise=add_obs_noise,
                         use_obs_estimator=use_obs_estimator,
                         wait_puck=wait_puck)

        self.traj_opt = TrajectoryOptimizer(self.env_info)

        if self.dof == 3:
            self.prev_c_pos = np.array([0.65, 0., 0.1645])
            self.prev_c_vel = np.array([0, 0, 0])
        else:
            self.prev_c_pos = np.array([0.65, 0., 0.1000])
            self.prev_c_vel = np.array([0, 0, 0])

        if interpolation_order is not None:
            self.dt = 0.02

    def reset(self, **kwargs):
        if self.dof == 3:
            self.prev_c_pos = np.array([0.65, 0., 0.1645])
            self.prev_c_vel = np.zeros([0, 0, 0])
        else:
            self.prev_c_pos = np.array([0.65, 0., 0.1000])
            self.prev_c_vel = np.zeros([0, 0, 0])
        return super().reset(**kwargs)

    def optimize_traj(self, traj_c_pos, traj_c_vel):
        # boundary condition
        q_start = self.q_prev
        dq_start = self.dq_prev
        ddq_start = self.ddq_prev

        # solve optimization
        traj_c = np.hstack([traj_c_pos, traj_c_vel])
        success, traj_q_opt = self.traj_opt.optimize_trajectory(traj_c, q_start, dq_start, None)

        # fit cubic-spline
        t = np.linspace(0, traj_q_opt.shape[0], traj_q_opt.shape[0] + 1) * 0.02
        f = CubicSpline(t, np.vstack([q_start, traj_q_opt]), axis=0, bc_type=((1, dq_start), (2, ddq_start)))
        df = f.derivative(1)
        ddf = f.derivative(2)

        # save boundary condition
        self.q_prev = np.array(f(t[-1]))
        self.dq_prev = np.array(df(t[-1]))
        self.ddq_prev = np.array(ddf(t[-1]))

        return np.array(f(t[1:])), np.array(df(t[1:]))

    def check_traj_validity(self, action, traj_pos, traj_vel):
        sub_traj_length = self.check_traj_length[self.sub_traj_idx]
        sub_traj_c_pos = traj_pos[:sub_traj_length] if sub_traj_length != -1 else traj_pos[:]
        sub_traj_c_vel = traj_vel[:sub_traj_length] if sub_traj_length != -1 else traj_vel[:]
        self.sub_traj_idx += 1

        if self.check_traj:

            # check tau
            invalid_tau = False
            invalid_delay = False

            # check ee constr
            constr_ee = self.constr_ee
            invalid_ee = np.any(sub_traj_c_pos < constr_ee[:, 0]) or np.any(sub_traj_c_pos > constr_ee[:, 1])

            if invalid_tau or invalid_delay or invalid_ee:
                return False, sub_traj_c_pos, sub_traj_c_vel

        self.prev_c_pos = sub_traj_c_pos[-1]
        self.prev_c_vel = sub_traj_c_vel[-1]
        # optimize traj
        sub_traj_j_pos, sub_traj_j_vel = self.optimize_traj(sub_traj_c_pos, sub_traj_c_vel)
        return True, sub_traj_j_pos, sub_traj_j_vel

    def get_invalid_traj_penalty(self, action, traj_pos, traj_vel):
        # violate tau penalty
        violate_tau_penalty = 0

        # violate delay penalty
        violate_delay_penalty = 0

        sub_traj_length = self.check_traj_length[self.sub_traj_idx-1]
        sub_traj_c_pos = traj_pos[:sub_traj_length] if sub_traj_length != -1 else traj_pos[:]
        sub_traj_c_vel = traj_vel[:sub_traj_length] if sub_traj_length != -1 else traj_vel[:]

        # violate ee penalty
        constr_ee = self.constr_ee
        num_violate_ee_constr = np.array((sub_traj_c_pos - constr_ee[:, 0] < 0), dtype=np.float32).mean() + \
                                np.array((sub_traj_c_pos - constr_ee[:, 1] > 0), dtype=np.float32).mean()
        max_violate_ee_constr = np.maximum(constr_ee[:, 0] - sub_traj_c_pos, 0).mean() + \
                                np.maximum(sub_traj_c_pos - constr_ee[:, 1], 0).mean()
        violate_ee_penalty = num_violate_ee_constr + max_violate_ee_constr

        traj_invalid_penalty = violate_tau_penalty + violate_delay_penalty + violate_ee_penalty
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

        sub_traj_length = self.check_traj_length[self.sub_traj_idx-1]
        sub_traj_length = self.horizon if sub_traj_length == -1 else sub_traj_length
        for k, v in info.items():
            info[k] = [v] * sub_traj_length

        info['trajectory_length'] = 1

        return obs, self.get_invalid_traj_penalty(action, traj_pos, traj_vel), True, info


class AirHockey3DofHitCart(AirHockeyGymHitCart):
    def __init__(self, interpolation_order=3, custom_reward_function='HitSparseRewardV0',
                 check_step=True, check_step_stop=False,
                 check_traj=True, check_traj_length=-1,
                 add_obs_noise: bool = False, use_obs_estimator: bool = False,
                 wait_puck=False):
        super().__init__(env_id="3dof-hit",
                         interpolation_order=interpolation_order,
                         custom_reward_function=custom_reward_function,
                         check_step=check_step,
                         check_step_stop=check_step_stop,
                         check_traj=check_traj,
                         check_traj_length=check_traj_length,
                         add_obs_noise=add_obs_noise,
                         use_obs_estimator=use_obs_estimator,
                         wait_puck=wait_puck)


class AirHockey7DofHitCart(AirHockeyGymHitCart):
    def __init__(self, interpolation_order=3, custom_reward_function='HitSparseRewardV0',
                 check_step=True, check_step_stop=False,
                 check_traj=True, check_traj_length=-1,
                 add_obs_noise: bool = False, use_obs_estimator: bool = False,
                 wait_puck=False):
        super().__init__(env_id="7dof-hit",
                         interpolation_order=interpolation_order,
                         custom_reward_function=custom_reward_function,
                         check_step=check_step,
                         check_step_stop=check_step_stop,
                         check_traj=check_traj,
                         check_traj_length=check_traj_length,
                         add_obs_noise=add_obs_noise,
                         use_obs_estimator=use_obs_estimator,
                         wait_puck=wait_puck)
