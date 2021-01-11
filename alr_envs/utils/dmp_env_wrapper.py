from mp_lib.phase import ExpDecayPhaseGenerator
from mp_lib.basis import DMPBasisGenerator
from mp_lib import dmps
import numpy as np
import gym


class DmpEnvWrapperBase(gym.Wrapper):
    def __init__(self, env, num_dof, num_basis, duration=1, dt=0.01, learn_goal=False):
        super(DmpEnvWrapperBase, self).__init__(env)
        self.num_dof = num_dof
        self.num_basis = num_basis
        self.dim = num_dof * num_basis
        if learn_goal:
            self.dim += num_dof
        self.learn_goal = True
        duration = duration   # seconds
        time_steps = int(duration / dt)
        self.t = np.linspace(0, duration, time_steps)

        phase_generator = ExpDecayPhaseGenerator(alpha_phase=5, duration=duration)
        basis_generator = DMPBasisGenerator(phase_generator, duration=duration, num_basis=self.num_basis)

        self.dmp = dmps.DMP(num_dof=num_dof,
                            basis_generator=basis_generator,
                            phase_generator=phase_generator,
                            num_time_steps=time_steps,
                            dt=dt
                            )

        self.dmp.dmp_start_pos = env.start_pos.reshape((1, num_dof))

        dmp_weights = np.zeros((num_basis, num_dof))
        dmp_goal_pos = np.zeros(num_dof)

        self.dmp.set_weights(dmp_weights, dmp_goal_pos)

    def goal_and_weights(self, params):
        if len(params.shape) > 1:
            assert params.shape[1] == self.dim
        else:
            assert len(params) == self.dim
            params = np.reshape(params, [1, self.dim])

        if self.learn_goal:
            goal_pos = params[0, -self.num_dof:]
            weight_matrix = np.reshape(params[:, :-self.num_dof], [self.num_basis, self.num_dof])
        else:
            goal_pos = None
            weight_matrix = np.reshape(params, [self.num_basis, self.num_dof])

        return goal_pos, weight_matrix

    def step(self, action, render=False):
        """ overwrite step function where action now is the weights and possible goal position"""
        raise NotImplementedError


class DmpEnvWrapperAngle(DmpEnvWrapperBase):
    def step(self, action, render=False):
        goal_pos, weight_matrix = self.goal_and_weights(action)
        self.dmp.set_weights(weight_matrix, goal_pos)
        trajectory, velocities = self.dmp.reference_trajectory(self.t)

        rews = []

        for t, traj in enumerate(trajectory):
            obs, rew, done, info = self.env.step(traj)
            rews.append(rew)
            if render:
                self.env.render(mode="human")
            if done:
                break

        reward = np.sum(rews)
        done = True
        info = {}

        return obs, reward, done, info


class DmpEnvWrapperVel(DmpEnvWrapperBase):
    def step(self, action, render=False):
        goal_pos, weight_matrix = self.goal_and_weights(action)
        weight_matrix *= 50
        self.dmp.set_weights(weight_matrix, goal_pos)
        trajectory, velocities = self.dmp.reference_trajectory(self.t)

        rews = []

        for t, vel in enumerate(velocities):
            obs, rew, done, info = self.env.step(vel)
            rews.append(rew)
            if render:
                self.env.render(mode="human")
            if done:
                break

        reward = np.sum(rews)
        info = {}

        return obs, reward, done, info
