import numpy as np
from typing import Union, Tuple
from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper
from fancy_gym.utils.time_aware_observation import TimeAwareObservation


class AirHockeyMPWrapper(RawInterfaceWrapper):

    @property
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        # q_pos, _ = self.env.env.get_joints(obs)
        return self.unwrapped.base_env.q_pos_prev
        # return self.unwrapped.robot_data.qpos.copy()
        # return self.unwrapped._data.qpos[:self.dof].copy()
        # return np.array([0, 0, 0])

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        # _, q_vel = self.env.env.get_joints(obs)
        return self.unwrapped.base_env.q_vel_prev
        # return self.unwrapped.robot_data.qvel.copy()
        # return self.unwrapped._data.qvel[:self.dof].copy()
        # return np.array([0, 0, 0])

    def set_episode_arguments(self, action, pos_traj, vel_traj):
        if self.interpolation_order is None:
            return pos_traj.reshape([-1, 20, self.dof]).copy(), pos_traj.reshape([-1, 20, self.dof]).copy()

        if self.dt == 0.001:
            return pos_traj[19::20].copy(), vel_traj[19::20].copy()
        return pos_traj, vel_traj

    def preprocessing_and_validity_callback(self, action: np.ndarray, pos_traj: np.ndarray, vel_traj: np.ndarray):
        return self.check_traj_validity(action, pos_traj, vel_traj)

    def invalid_traj_callback(self, action: np.ndarray, pos_traj: np.ndarray, vel_traj: np.ndarray,
                              return_contextual_obs):
        obs, trajectory_return, done, infos = self.get_invalid_traj_return(action, pos_traj, vel_traj)
        if issubclass(self.env.__class__, TimeAwareObservation):
            obs = np.append(obs, np.array([0], dtype=obs.dtype))
        return obs, trajectory_return, done, infos


class AirHockey3DofHitMPWrapper(AirHockeyMPWrapper):
    @property
    def context_mask(self) -> np.ndarray:
        return np.hstack([
            [True, True, True],  # puck position [x, y, theta]
            [False] * 3,  # puck velocity [dx, dy, dtheta]
            [False] * 3,  # joint position
            [False] * 3,  # joint velocity
        ])


class AirHockey7DofHitMPWrapper(AirHockeyMPWrapper):
    @property
    def context_mask(self) -> np.ndarray:
        return np.hstack([
            [True] * 23
        ])

    @property
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.unwrapped.base_env.q_pos_prev[:7]


    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.unwrapped.base_env.q_vel_prev[:7]


class DefendMPWrapper(AirHockeyMPWrapper):

    @property
    def context_mask(self) -> np.ndarray:
        return np.hstack([
            [True, True, True],  # puck position [x, y, theta]
            [True, True, True],  # puck velocity [dx, dy, dtheta]
            [False] * 3,  # joint position
            [False] * 3,  # joint velocity
        ])
