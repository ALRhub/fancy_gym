import numpy as np
from typing import Union, Tuple
from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper
from fancy_gym.utils.time_aware_observation import TimeAwareObservation

from air_hockey_challenge.utils import world_to_robot, robot_to_world


class AirHockeyCartMPWrapper(RawInterfaceWrapper):
    @property
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        ee_pos_world = self.unwrapped.base_env.get_ee()[0]
        ee_pos_robot = world_to_robot(self.env_info["robot"]["base_frame"][0], ee_pos_world)[0]
        return ee_pos_robot[:2]
        # return np.array([6.49932e-01, 2.05280e-05, 1.00000e-01])[:2]

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        ee_vel_world = self.unwrapped.base_env.get_ee()[1][3:]
        ee_vel_robot = ee_vel_world
        return ee_vel_robot[:2]
        # return np.array([0, 0, 0])[:2]

    @property
    def current_des_pos(self):
        return self.prev_c_pos[:2]

    @property
    def current_des_vel(self):
        return self.prev_c_vel[:2]

    def set_episode_arguments(self, action, pos_traj, vel_traj):
        if self.dof == 3:
            pos_traj = np.hstack([pos_traj, 1.0e-1 * np.ones([pos_traj.shape[0], 1])])
            vel_traj = np.hstack([vel_traj, np.zeros([vel_traj.shape[0], 1])])
        else:
            pos_traj = np.hstack([pos_traj, 1.645e-1 * np.ones([pos_traj.shape[0], 1])])
            vel_traj = np.hstack([vel_traj, np.zeros([vel_traj.shape[0], 1])])
        if self.dt == 0.001:
            return pos_traj[19::20].copy(), vel_traj[19::20].copy()
        return pos_traj.copy(), vel_traj.copy()

    def preprocessing_and_validity_callback(self, action: np.ndarray, pos_traj: np.ndarray, vel_traj: np.ndarray):
        return self.check_traj_validity(action, pos_traj, vel_traj)

    def invalid_traj_callback(self, action: np.ndarray, pos_traj: np.ndarray, vel_traj: np.ndarray,
                              return_contextual_obs=True):
        obs, trajectory_return, done, infos = self.get_invalid_traj_return(action, pos_traj, vel_traj)
        if issubclass(self.env.__class__, TimeAwareObservation):
            obs = np.append(obs, np.array([0], dtype=obs.dtype))
        return obs, trajectory_return, done, infos


class AirHockey3DofHitCartMPWrapper(AirHockeyCartMPWrapper):
    @property
    def context_mask(self) -> np.ndarray:
        return np.hstack([
            [True, True, True],  # puck position [x, y, theta]
            [False] * 3,  # puck velocity [dx, dy, dtheta]
            [False] * 3,  # joint position
            [False] * 3,  # joint velocity
        ])


class AirHockey7DofHitCartMPWrapper(AirHockeyCartMPWrapper):
    @property
    def context_mask(self) -> np.ndarray:
        return np.hstack([
            [True] * 23
        ])
