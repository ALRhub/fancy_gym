import gym
import numpy as np
from mp_lib import det_promp

from alr_envs.utils.mps.alr_env import AlrEnv
from alr_envs.utils.mps.mp_wrapper import MPWrapper


class DetPMPWrapper(MPWrapper):
    def __init__(self, env: AlrEnv, num_dof, num_basis, width, start_pos=None, duration=1, post_traj_time=0.,
                 policy_type=None, weights_scale=1, zero_start=False, zero_goal=False, learn_mp_length: bool =True,
                 **mp_kwargs):
        self.duration = duration  # seconds

        super().__init__(env=env, num_dof=num_dof, duration=duration, post_traj_time=post_traj_time,
                         policy_type=policy_type, weights_scale=weights_scale, num_basis=num_basis,
                         width=width, zero_start=zero_start, zero_goal=zero_goal,
                         **mp_kwargs)

        self.learn_mp_length = learn_mp_length
        if self.learn_mp_length:
            parameter_space_shape = (1+num_basis*num_dof,)
        else:
            parameter_space_shape = (num_basis * num_dof,)
        self.min_param = -np.inf
        self.max_param = np.inf
        self.parameterization_space = gym.spaces.Box(low=self.min_param, high=self.max_param,
                                                     shape=parameter_space_shape, dtype=np.float32)

        self.start_pos = start_pos

    def initialize_mp(self, num_dof: int, duration: int, num_basis: int = 5, width: float = None,
                      zero_start: bool = False, zero_goal: bool = False, **kwargs):
        pmp = det_promp.DeterministicProMP(n_basis=num_basis, n_dof=num_dof, width=width, off=0.01,
                                           zero_start=zero_start, zero_goal=zero_goal)

        weights = np.zeros(shape=(num_basis, num_dof))
        pmp.set_weights(duration, weights)

        return pmp

    def mp_rollout(self, action):
        if self.learn_mp_length:
            duration = max(1, self.duration*abs(action[0]))
            params = np.reshape(action[1:], (self.mp.n_basis, -1)) * self.weights_scale # TODO: Fix Bug when zero_start is true
        else:
            duration = self.duration
            params = np.reshape(action, (self.mp.n_basis, -1)) * self.weights_scale # TODO: Fix Bug when zero_start is true
        self.mp.set_weights(1., params)
        _, des_pos, des_vel, _ = self.mp.compute_trajectory(frequency=max(1, duration))
        if self.mp.zero_start:
            des_pos += self.start_pos

        return des_pos, des_vel
