from alr_envs.mp.episodic_wrapper import EpisodicWrapper
from typing import Union, Tuple
import numpy as np
import gym


class NewMPWrapper(EpisodicWrapper):
    @property
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.env.sim.data.qpos[0:7].copy()

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.env.sim.data.qvel[0:7].copy()

    def set_active_obs(self):
        return np.hstack([
            [False] * 7,  # cos
            [False] * 7,  # sin
            [True] * 2,  # xy position of cup
            [False]  # env steps
        ])

    def _step_callback(self, t: int, env_spec_params: Union[np.ndarray, None], step_action: np.ndarray) -> Union[np.ndarray]:
        if self.env.learn_release_step:
            return np.concatenate((step_action, np.atleast_1d(env_spec_params)))
        else:
            return step_action

    def _episode_callback(self, action: np.ndarray) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
        if self.env.learn_release_step:
            return action[:-1], action[-1]  # mp_params, release step
        else:
            return action, None

    def set_action_space(self):
        if self.env.learn_release_step:
            min_action_bounds, max_action_bounds = self.mp.get_param_bounds()
            min_action_bounds = np.concatenate((min_action_bounds.numpy(), [self.env.action_space.low[-1]]))
            max_action_bounds = np.concatenate((max_action_bounds.numpy(), [self.env.action_space.high[-1]]))
            self.mp_action_space = gym.spaces.Box(low=min_action_bounds, high=max_action_bounds, dtype=np.float32)
            return self.mp_action_space
        else:
            return super(NewMPWrapper, self).set_action_space()
