from typing import Union, Tuple

import gym
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

    def episode_callback(self, action: np.ndarray, traj_gen: MPInterface) -> Tuple[
        np.ndarray, Union[np.ndarray, None]]:
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
        return action, None
