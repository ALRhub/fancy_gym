import numpy as np
from typing import Union, Tuple

from .air_hockey_hit import AirHockey3DofHit, AirHockey7DofHit
from .air_hockey_utils import TrajectoryOptimizer
from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper
from fancy_gym.utils.time_aware_observation import TimeAwareObservation
from scipy.interpolate import CubicSpline


class AirHockeyPlanarHitCart(AirHockey3DofHit):
    def __init__(self, dt=0.02, reward_function='HitSparseRewardV0', replan_steps=-1):
        super().__init__(dt, reward_function, replan_steps)

        self.traj_opt = TrajectoryOptimizer(self.env_info)

    def check_traj_validity(self, action, traj_pos, traj_vel):
        # check tau
        invalid_tau = False
        if action.shape[0] % 2 != 0:
            tau_bound = [1.5, 3.0]
            invalid_tau = action[0] < tau_bound[0] or action[0] > tau_bound[1]

        if self.replan_steps != -1:
            valid_pos = traj_pos[:self.replan_steps]
            valid_vel = traj_vel[:self.replan_steps]
        else:
            valid_pos = traj_pos
            valid_vel = traj_vel

        # check ee constr
        constr_ee = np.array([[+0.585, +1.585], [-0.470, +0.470], [+0.080, +0.120]])
        invalid_ee = np.any(valid_pos < constr_ee[:, 0]) or np.any(valid_pos > constr_ee[:, 1])

        if invalid_tau or invalid_ee:
            return False, traj_pos, traj_vel

        # optimize traj
        q_start = np.array([-1.15570, +1.30024, +1.44280])
        dq_start = np.array([0, 0, 0])
        success, traj_q_opt = self.traj_opt.optimize_trajectory(np.hstack([valid_pos, valid_vel]), q_start, dq_start, None)
        t = np.linspace(0, traj_q_opt.shape[0], traj_q_opt.shape[0] + 1) * 0.02
        f = CubicSpline(t, np.vstack([q_start, traj_q_opt]), axis=0, bc_type=((1, dq_start),
                                                                              (2, np.zeros_like(dq_start))))
        df = f.derivative(1)
        traj_pos = np.array(f(t[1:]))
        traj_vel = np.array(df(t[1:]))
        return True, traj_pos, traj_vel

    def get_invalid_traj_penalty(self, action, traj_pos, traj_vel):
        # violate tau penalty
        violate_tau_penalty = 0
        if action.shape[0] % 2 != 0:
            tau_bound = [1.5, 3.0]
            violate_tau_penalty = np.max([0, action[0] - tau_bound[1]]) + np.max([0, tau_bound[0] - action[0]])

        if self.replan_steps != -1:
            valid_pos = traj_pos[:self.replan_steps]
            valid_vel = traj_vel[:self.replan_steps]
        else:
            valid_pos = traj_pos
            valid_vel = traj_vel

        # violate ee penalty
        constr_ee = np.array([[+0.585, +1.585], [-0.470, +0.470], [+0.080, +0.120]])
        num_violate_ee_constr = np.array((valid_pos - constr_ee[:, 0] < 0), dtype=np.float32).mean() + \
                                np.array((valid_pos - constr_ee[:, 1] > 0), dtype=np.float32).mean()
        max_violate_ee_constr = np.maximum(constr_ee[:, 0] - valid_pos, 0).mean() + \
                                np.maximum(valid_pos - constr_ee[:, 1], 0).mean()
        violate_ee_penalty = num_violate_ee_constr + max_violate_ee_constr

        traj_invalid_penalty = violate_tau_penalty + violate_ee_penalty
        return -3 * traj_invalid_penalty


class HitCartMPWrapper(RawInterfaceWrapper):

    @property
    def context_mask(self) -> np.ndarray:
        return np.hstack([
            [True, True, True],  # puck position [x, y, theta]
            [False] * 3,  # puck velocity [dx, dy, dtheta]
            [False] * 3,  # joint position
            [False] * 3,  # joint velocity
        ])

    @property
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        # return self.unwrapped.base_env.get_ee()[0][:2]
        return np.array([6.49932e-01, 2.05280e-05, 1.00000e-01])[:2]

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        # return self.unwrapped.base_env.get_ee()[1][:2]
        return np.array([0, 0, 0])[:2]

    def set_episode_arguments(self, action, pos_traj, vel_traj):
        pos_traj = np.hstack([pos_traj, 1e-1 * np.ones([pos_traj.shape[0], 1])])
        vel_traj = np.hstack([vel_traj, np.zeros([vel_traj.shape[0], 1])])
        if self.dt == 0.001:
            return pos_traj[19::20].copy(), vel_traj[19::20].copy()
        return pos_traj.copy(), vel_traj.copy()

    def preprocessing_and_validity_callback(self, action: np.ndarray, pos_traj: np.ndarray, vel_traj: np.ndarray):
        return self.check_traj_validity(action, pos_traj, vel_traj)

    def invalid_traj_callback(self, action: np.ndarray, pos_traj: np.ndarray, vel_traj: np.ndarray,
                              return_contextual_obs):
        obs, trajectory_return, done, infos = self.get_invalid_traj_return(action, pos_traj, vel_traj)
        if issubclass(self.env.__class__, TimeAwareObservation):
            obs = np.append(obs, np.array([0], dtype=obs.dtype))
        return obs, trajectory_return, done, infos