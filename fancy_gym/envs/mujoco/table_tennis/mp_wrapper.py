from typing import Union, Tuple

import numpy as np

from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper
from fancy_gym.envs.mujoco.table_tennis.table_tennis_utils import jnt_pos_low, jnt_pos_high, delay_bound, tau_bound


class TT_MPWrapper(RawInterfaceWrapper):
    mp_config = {
        'ProMP': {
            'phase_generator_kwargs': {
                'learn_tau': False,
                'learn_delay': False,
                'tau_bound': [0.8, 1.5],
                'delay_bound': [0.05, 0.15],
            },
            'controller_kwargs': {
                'p_gains': 0.5 * np.array([1.0, 4.0, 2.0, 4.0, 1.0, 4.0, 1.0]),
                'd_gains': 0.5 * np.array([0.1, 0.4, 0.2, 0.4, 0.1, 0.4, 0.1]),
            },
            'basis_generator_kwargs': {
                'num_basis': 3,
                'num_basis_zero_start': 1,
                'num_basis_zero_goal': 1,
            },
            'black_box_kwargs': {
                'verbose': 2,
            },
        },
        'DMP': {},
        'ProDMP': {
            'phase_generator_kwargs': {
                'learn_tau': True,
                'learn_delay': True,
                'tau_bound': [0.8, 1.5],
                'delay_bound': [0.05, 0.15],
                'alpha_phase': 3,
            },
            'controller_kwargs': {
                'p_gains': 0.5 * np.array([1.0, 4.0, 2.0, 4.0, 1.0, 4.0, 1.0]),
                'd_gains': 0.5 * np.array([0.1, 0.4, 0.2, 0.4, 0.1, 0.4, 0.1]),
            },
            'basis_generator_kwargs': {
                'num_basis': 3,
                'alpha': 25,
                'basis_bandwidth_factor': 3,
            },
            'trajectory_generator_kwargs': {
                'weights_scale': 0.7,
                'auto_scale_basis': True,
                'relative_goal': True,
                'disable_goal': True,
            },
        },
    }

    # Random x goal + random init pos
    @property
    def context_mask(self):
        return np.hstack([
            [False] * 7,  # joints position
            [False] * 7,  # joints velocity
            [True] * 2,  # position ball x, y
            [False] * 1,  # position ball z
            # [True] * 3,    # velocity ball x, y, z
            [True] * 2,  # target landing position
            # [True] * 1,  # time
        ])

    @property
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.data.qpos[:7].copy()

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.data.qvel[:7].copy()

    def preprocessing_and_validity_callback(self, action: np.ndarray, pos_traj: np.ndarray, vel_traj: np.ndarray,
                                            tau_bound: list, delay_bound:list):
        return self.check_traj_validity(action, pos_traj, vel_traj, tau_bound, delay_bound)

    def set_episode_arguments(self, action, pos_traj, vel_traj):
        return pos_traj, vel_traj

    def invalid_traj_callback(self, action: np.ndarray, pos_traj: np.ndarray, vel_traj: np.ndarray,
                              return_contextual_obs: bool, tau_bound:list, delay_bound:list) -> Tuple[np.ndarray, float, bool, dict]:
        return self.get_invalid_traj_step_return(action, pos_traj, return_contextual_obs, tau_bound, delay_bound)


class TT_MPWrapper_Replan(TT_MPWrapper):
    mp_config = {
        'ProMP': {},
        'DMP': {},
        'ProDMP': {
            'phase_generator_kwargs': {
                'learn_tau': True,
                'learn_delay': True,
                'tau_bound': [0.8, 1.5],
                'delay_bound': [0.05, 0.15],
                'alpha_phase': 3,
            },
            'controller_kwargs': {
                'p_gains': 0.5 * np.array([1.0, 4.0, 2.0, 4.0, 1.0, 4.0, 1.0]),
                'd_gains': 0.5 * np.array([0.1, 0.4, 0.2, 0.4, 0.1, 0.4, 0.1]),
            },
            'basis_generator_kwargs': {
                'num_basis': 2,
                'alpha': 25,
                'basis_bandwidth_factor': 3,
            },
            'trajectory_generator_kwargs': {
                'auto_scale_basis': True,
                'goal_offset': 1.0,
            },
            'black_box_kwargs': {
                'max_planning_times': 3,
                'replanning_schedule': lambda pos, vel, obs, action, t: t % 50 == 0,
            },
        },
    }


class TTVelObs_MPWrapper(TT_MPWrapper):
    # Will inherit mp_config from TT_MPWrapper

    @property
    def context_mask(self):
        return np.hstack([
            [False] * 7,  # joints position
            [False] * 7,  # joints velocity
            [True] * 2,  # position ball x, y
            [False] * 1,  # position ball z
            [True] * 3,    # velocity ball x, y, z
            [True] * 2,  # target landing position
            # [True] * 1,  # time
        ])


class TTVelObs_MPWrapper_Replan(TT_MPWrapper_Replan):
    # Will inherit mp_config from TT_MPWrapper_Replan

    @property
    def context_mask(self):
        return np.hstack([
            [False] * 7,  # joints position
            [False] * 7,  # joints velocity
            [True] * 2,  # position ball x, y
            [False] * 1,  # position ball z
            [True] * 3,    # velocity ball x, y, z
            [True] * 2,  # target landing position
            # [True] * 1,  # time
        ])

class TTRndRobot_MPWrapper(TT_MPWrapper):
    @property
    def context_mask(self):
        return np.hstack([
            [True] * 7,  # joints position
            [False] * 7,  # joints velocity
            [True] * 2,  # position ball x, y
            [False] * 1,  # position ball z
            [True] * 2,  # target landing position
            # [True] * 1,  # time
        ])