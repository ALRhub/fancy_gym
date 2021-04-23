from mp_lib.phase import ExpDecayPhaseGenerator
from mp_lib.basis import DMPBasisGenerator
from mp_lib import dmps
import numpy as np
import gym

from alr_envs.utils.wrapper.mp_wrapper import MPWrapper


class DmpWrapper(MPWrapper):

    def __init__(self, env: gym.Env, num_dof: int, num_basis: int, start_pos: np.ndarray = None,
                 final_pos: np.ndarray = None, duration: int = 1, alpha_phase: float = 2., dt: float = None,
                 learn_goal: bool = False, return_to_start: bool = False, post_traj_time: float = 0.,
                 weights_scale: float = 1., goal_scale: float = 1., bandwidth_factor: float = 3.,
                 policy_type: str = None):

        """
        This Wrapper generates a trajectory based on a DMP and will only return episodic performances.
        Args:
            env:
            num_dof:
            num_basis:
            start_pos:
            final_pos:
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
        dt = env.dt if hasattr(env, "dt") else dt
        assert dt is not None
        start_pos = start_pos if start_pos is not None else env.start_pos if hasattr(env, "start_pos") else None
        assert start_pos is not None
        if learn_goal:
            final_pos = np.zeros_like(start_pos)  # arbitrary, will be learned
        else:
            final_pos = final_pos if final_pos is not None else start_pos if return_to_start else None
        assert final_pos is not None
        self.t = np.linspace(0, duration, int(duration / dt))
        self.goal_scale = goal_scale

        super().__init__(env, num_dof, duration, dt, post_traj_time, policy_type, weights_scale,
                         num_basis=num_basis, start_pos=start_pos, final_pos=final_pos, alpha_phase=alpha_phase,
                         bandwidth_factor=bandwidth_factor)

        action_bounds = np.inf * np.ones((np.prod(self.mp.dmp_weights.shape) + (num_dof if learn_goal else 0)))
        self.action_space = gym.spaces.Box(low=-action_bounds, high=action_bounds, dtype=np.float32)

    def initialize_mp(self, num_dof: int, duration: int, dt: float, num_basis: int = 5, start_pos: np.ndarray = None,
                      final_pos: np.ndarray = None, alpha_phase: float = 2., bandwidth_factor: float = 3.):

        phase_generator = ExpDecayPhaseGenerator(alpha_phase=alpha_phase, duration=duration)
        basis_generator = DMPBasisGenerator(phase_generator, duration=duration, num_basis=num_basis,
                                            basis_bandwidth_factor=bandwidth_factor)

        dmp = dmps.DMP(num_dof=num_dof, basis_generator=basis_generator, phase_generator=phase_generator,
                       num_time_steps=int(duration / dt), dt=dt)

        dmp.dmp_start_pos = start_pos.reshape((1, num_dof))

        weights = np.zeros((num_basis, num_dof))
        goal_pos = np.zeros(num_dof) if self.learn_goal else final_pos

        dmp.set_weights(weights, goal_pos)
        return dmp

    def goal_and_weights(self, params):
        assert params.shape[-1] == self.action_space.shape[0]
        params = np.atleast_2d(params)

        if self.learn_goal:
            goal_pos = params[0, -self.mp.num_dimensions:]  # [num_dof]
            params = params[:, :-self.mp.num_dimensions]  # [1,num_dof]
            # weight_matrix = np.reshape(params[:, :-self.num_dof], [self.num_basis, self.num_dof])
        else:
            goal_pos = self.mp.dmp_goal_pos.flatten()
            assert goal_pos is not None
            # weight_matrix = np.reshape(params, [self.num_basis, self.num_dof])

        weight_matrix = np.reshape(params, self.mp.dmp_weights.shape)
        return goal_pos * self.goal_scale, weight_matrix * self.weights_scale

    def mp_rollout(self, action):
        goal_pos, weight_matrix = self.goal_and_weights(action)
        self.mp.set_weights(weight_matrix, goal_pos)
        return self.mp.reference_trajectory(self.t)
