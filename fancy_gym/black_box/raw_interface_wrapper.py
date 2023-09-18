from typing import Union, Tuple

import gymnasium as gym
import numpy as np
from mp_pytorch.mp.mp_interfaces import MPInterface


class RawInterfaceWrapper(gym.Wrapper):

    @property
    def context_mask(self) -> np.ndarray:
        """
        Returns boolean mask of the same shape as the observation space.
        It determines whether the observation is returned for the contextual case or not.
        This effectively allows to filter unwanted or unnecessary observations from the full step-based case.
        E.g. Velocities starting at 0 are only changing after the first action. Given we only receive the
        context/part of the first observation, the velocities are not necessary in the observation for the task.
        Returns:
            bool array representing the indices of the observations

        """
        return np.ones(self.env.observation_space.shape[0], dtype=bool)

    @property
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        """
            Returns the current position of the action/control dimension.
            The dimensionality has to match the action/control dimension.
            This is not required when exclusively using velocity control,
            it should, however, be implemented regardless.
            E.g. The joint positions that are directly or indirectly controlled by the action.
        """
        raise NotImplementedError

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        """
            Returns the current velocity of the action/control dimension.
            The dimensionality has to match the action/control dimension.
            This is not required when exclusively using position control,
            it should, however, be implemented regardless.
            E.g. The joint velocities that are directly or indirectly controlled by the action.
        """
        raise NotImplementedError

    @property
    def dt(self) -> float:
        """
        Control frequency of the environment
        Returns: float

        """
        return self.env.dt

    def preprocessing_and_validity_callback(self, action: np.ndarray, pos_traj: np.ndarray, vel_traj: np.ndarray,
                                            tau_bound: list = None, delay_bound: list = None ) \
            -> Tuple[bool, np.ndarray, np.ndarray]:
        """
        Used to preprocess the action and check if the desired trajectory is valid.
        Args:
            action:  a vector instance of the whole action space, includes traj_gen parameters and additional parameters if
            specified, else only traj_gen parameters
            pos_traj: a vector instance of the raw position trajectory
            vel_traj: a vector instance of the raw velocity trajectory
            tau_bound: a list of two elements, the lower and upper bound of the trajectory length scaling factor
            delay_bound: a list of two elements, the lower and upper bound of the time to wait before execute
        Returns:
            validity flag: bool, True if the raw trajectory is valid, False if not
            pos_traj: a vector instance of the preprocessed position trajectory 
            vel_traj: a vector instance of the preprocessed velocity trajectory
        """
        return True, pos_traj, vel_traj

    def set_episode_arguments(self, action, pos_traj, vel_traj):
        """
        Used to set the arguments for env that valid for the whole episode
        deprecated, replaced by preprocessing_and_validity_callback
        Args:
            action:  a vector instance of the whole action space, includes traj_gen parameters and additional parameters if
            specified, else only traj_gen parameters
            pos_traj: a vector instance of the raw position trajectory
            vel_traj: a vector instance of the raw velocity trajectory
        Returns:
            pos_traj: a vector instance of the preprocessed position trajectory
            vel_traj: a vector instance of the preprocessed velocity trajectory
        """
        return pos_traj, vel_traj

    def episode_callback(self, action: np.ndarray, pos_traj: np.ndarray, vel_traj: np.array) -> Tuple[bool]:
        """
        Used to extract the parameters for the movement primitive and other parameters from an action array which might
        include other actions like ball releasing time for the beer pong environment.
        This only needs to be overwritten if the action space is modified.
        Args:
            action: a vector instance of the whole action space, includes traj_gen parameters and additional parameters if
            specified, else only traj_gen parameters

        Returns:
            Tuple: mp_arguments and other arguments
        """
        return True

    def invalid_traj_callback(self, action: np.ndarray, pos_traj: np.ndarray, vel_traj: np.ndarray,
                              tau_bound: list, delay_bound: list) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Used to return a artificial return from the env if the desired trajectory is invalid.
        Args:
            action:  a vector instance of the whole action space, includes traj_gen parameters and additional parameters if
            specified, else only traj_gen parameters
            pos_traj: a vector instance of the raw position trajectory
            vel_traj: a vector instance of the raw velocity trajectory
            tau_bound: a list of two elements, the lower and upper bound of the trajectory length scaling factor
            delay_bound: a list of two elements, the lower and upper bound of the time to wait before execute
        Returns:
            obs: artificial observation if the trajectory is invalid, by default a zero vector
            reward: artificial reward if the trajectory is invalid, by default 0
            terminated: artificial terminated if the trajectory is invalid, by default True
            truncated: artificial truncated if the trajectory is invalid, by default False
            info: artificial info if the trajectory is invalid, by default empty dict
        """
        return np.zeros(1), 0, True, False, {}
