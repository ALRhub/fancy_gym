import gym
import numpy as np
from mp_lib import det_promp

from alr_envs.utils.wrapper.mp_wrapper import MPWrapper


class DetPMPWrapper(MPWrapper):
    def __init__(self, env, num_dof, num_basis, width, start_pos=None, duration=1, dt=0.01, post_traj_time=0.,
                 policy_type=None, weights_scale=1, zero_start=False, zero_goal=False, **mp_kwargs):
        # self.duration = duration  # seconds

        super().__init__(env, num_dof, duration, dt, post_traj_time, policy_type, weights_scale,
                         num_basis=num_basis, width=width, start_pos=start_pos, zero_start=zero_start,
                         zero_goal=zero_goal)

        action_bounds = np.inf * np.ones((self.mp.n_basis * self.mp.n_dof))
        self.action_space = gym.spaces.Box(low=-action_bounds, high=action_bounds, dtype=np.float32)

        self.start_pos = start_pos
        self.dt = dt

    def initialize_mp(self, num_dof: int, duration: int, dt: float, num_basis: int = 5, width: float = None,
                      start_pos: np.ndarray = None, zero_start: bool = False, zero_goal: bool = False):
        pmp = det_promp.DeterministicProMP(n_basis=num_basis, n_dof=num_dof, width=width, off=0.01,
                                           zero_start=zero_start, zero_goal=zero_goal)

        weights = np.zeros(shape=(num_basis, num_dof))
        pmp.set_weights(duration, weights)

        return pmp

    def mp_rollout(self, action):
        params = np.reshape(action, (self.mp.n_basis, self.mp.n_dof)) * self.weights_scale
        self.mp.set_weights(self.duration, params)
        _, des_pos, des_vel, _ = self.mp.compute_trajectory(1 / self.dt, 1.)
        if self.mp.zero_start:
            des_pos += self.start_pos[None, :]

        return des_pos, des_vel
