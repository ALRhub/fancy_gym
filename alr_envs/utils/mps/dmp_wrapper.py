import gym
import numpy as np
from mp_lib import dmps
from mp_lib.basis import DMPBasisGenerator
from mp_lib.phase import ExpDecayPhaseGenerator

from alr_envs.utils.mps.alr_env import AlrEnv
from alr_envs.utils.mps.mp_wrapper import MPWrapper


class DmpWrapper(MPWrapper):

    def __init__(self, env: AlrEnv, num_dof: int, num_basis: int,
                 duration: int = 1, alpha_phase: float = 2., dt: float = None,
                 learn_goal: bool = False, post_traj_time: float = 0.,
                 weights_scale: float = 1., goal_scale: float = 1., bandwidth_factor: float = 3.,
                 policy_type: str = None, render_mode: str = None):

        """
        This Wrapper generates a trajectory based on a DMP and will only return episodic performances.
        Args:
            env:
            num_dof:
            num_basis:
            duration:
            alpha_phase:
            dt:
            learn_goal:
            post_traj_time:
            policy_type:
            weights_scale:
            goal_scale:
        """
        self.learn_goal = learn_goal

        self.t = np.linspace(0, duration, int(duration / dt))
        self.goal_scale = goal_scale

        super().__init__(env=env, num_dof=num_dof, duration=duration, post_traj_time=post_traj_time,
                         policy_type=policy_type, weights_scale=weights_scale, render_mode=render_mode,
                         num_basis=num_basis, alpha_phase=alpha_phase, bandwidth_factor=bandwidth_factor)

        action_bounds = np.inf * np.ones((np.prod(self.mp.weights.shape) + (num_dof if learn_goal else 0)))
        self.action_space = gym.spaces.Box(low=-action_bounds, high=action_bounds, dtype=np.float32)

    def initialize_mp(self, num_dof: int, duration: int, num_basis: int, alpha_phase: float = 2.,
                      bandwidth_factor: int = 3, **kwargs):

        phase_generator = ExpDecayPhaseGenerator(alpha_phase=alpha_phase, duration=duration)
        basis_generator = DMPBasisGenerator(phase_generator, duration=duration, num_basis=num_basis,
                                            basis_bandwidth_factor=bandwidth_factor)

        dmp = dmps.DMP(num_dof=num_dof, basis_generator=basis_generator, phase_generator=phase_generator,
                       dt=self.dt)

        return dmp

    def goal_and_weights(self, params):
        assert params.shape[-1] == self.action_space.shape[0]
        params = np.atleast_2d(params)

        if self.learn_goal:
            goal_pos = params[0, -self.mp.num_dimensions:]  # [num_dof]
            params = params[:, :-self.mp.num_dimensions]  # [1,num_dof]
        else:
            goal_pos = self.env.goal_pos
            assert goal_pos is not None

        weight_matrix = np.reshape(params, self.mp.weights.shape)  # [num_basis, num_dof]
        return goal_pos * self.goal_scale, weight_matrix * self.weights_scale

    def mp_rollout(self, action):
        self.mp.dmp_start_pos = self.env.start_pos
        goal_pos, weight_matrix = self.goal_and_weights(action)
        self.mp.set_weights(weight_matrix, goal_pos)
        return self.mp.reference_trajectory(self.t)
