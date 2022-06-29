from typing import Union, Tuple

import gym
import numpy as np
from abc import abstractmethod


class RawInterfaceWrapper(gym.Wrapper):

    @property
    @abstractmethod
    def context_mask(self) -> np.ndarray:
        """
        This function defines the contexts. The contexts are defined as specific observations.
        Returns:
            bool array representing the indices of the observations

        """
        return np.ones(self.env.observation_space.shape[0], dtype=bool)

    @property
    @abstractmethod
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        """
            Returns the current position of the action/control dimension.
            The dimensionality has to match the action/control dimension.
            This is not required when exclusively using velocity control,
            it should, however, be implemented regardless.
            E.g. The joint positions that are directly or indirectly controlled by the action.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        """
            Returns the current velocity of the action/control dimension.
            The dimensionality has to match the action/control dimension.
            This is not required when exclusively using position control,
            it should, however, be implemented regardless.
            E.g. The joint velocities that are directly or indirectly controlled by the action.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def dt(self) -> float:
        """
        Control frequency of the environment
        Returns: float

        """

    def do_replanning(self, pos, vel, s, a, t):
        # return t % 100 == 0
        # return bool(self.replanning_model(s))
        return False

    def _episode_callback(self, action: np.ndarray) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
        """
        Used to extract the parameters for the motion primitive and other parameters from an action array which might
        include other actions like ball releasing time for the beer pong environment.
        This only needs to be overwritten if the action space is modified.
        Args:
            action: a vector instance of the whole action space, includes trajectory_generator parameters and additional parameters if
            specified, else only trajectory_generator parameters

        Returns:
            Tuple: mp_arguments and other arguments
        """
        return action, None

    def _step_callback(self, t: int, env_spec_params: Union[np.ndarray, None], step_action: np.ndarray) -> Union[
        np.ndarray]:
        """
        This function can be used to modify the step_action with additional parameters e.g. releasing the ball in the
        Beerpong env. The parameters used should not be part of the motion primitive parameters.
        Returns step_action by default, can be overwritten in individual mp_wrappers.
        Args:
            t: the current time step of the episode
            env_spec_params: the environment specific parameter, as defined in function _episode_callback
            (e.g. ball release time in Beer Pong)
            step_action: the current step-based action

        Returns:
            modified step action
        """
        return step_action
