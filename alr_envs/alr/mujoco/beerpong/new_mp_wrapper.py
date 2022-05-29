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

    # def set_mp_action_space(self):
    #     min_action_bounds, max_action_bounds = self.mp.get_param_bounds()
    #     if self.mp.learn_tau:
    #         min_action_bounds[0] = 20*self.env.dt
    #         max_action_bounds[0] = 260*self.env.dt
    #     mp_action_space = gym.spaces.Box(low=min_action_bounds.numpy(), high=max_action_bounds.numpy(),
    #                                      dtype=np.float32)
    #     return mp_action_space

    # def _step_callback(self, t: int, env_spec_params: Union[np.ndarray, None], step_action: np.ndarray) -> Union[np.ndarray]:
    #     if self.mp.learn_tau:
    #         return np.concatenate((step_action, np.atleast_1d(env_spec_params)))
    #     else:
    #         return step_action

    def _episode_callback(self, action: np.ndarray) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
        if self.mp.learn_tau:
            self.env.env.release_step = action[0]/self.env.dt         # Tau value
            return action, None
        else:
            return action, None

    def set_context(self, context):
        xyz = np.zeros(3)
        xyz[:2] = context
        xyz[-1] = 0.840
        self.env.env.model.body_pos[self.env.env.cup_table_id] = xyz
        return self.get_observation_from_step(self.env.env._get_obs())
    # def set_action_space(self):
    #     if self.mp.learn_tau:
    #         min_action_bounds, max_action_bounds = self.mp.get_param_bounds()
    #         min_action_bounds = np.concatenate((min_action_bounds.numpy(), [self.env.action_space.low[-1]]))
    #         max_action_bounds = np.concatenate((max_action_bounds.numpy(), [self.env.action_space.high[-1]]))
    #         self.action_space = gym.spaces.Box(low=min_action_bounds, high=max_action_bounds, dtype=np.float32)
    #         return self.action_space
    #     else:
    #         return super(NewMPWrapper, self).set_action_space()
