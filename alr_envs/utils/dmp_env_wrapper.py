from alr_envs.utils.policies import get_policy_class
from mp_lib.phase import ExpDecayPhaseGenerator
from mp_lib.basis import DMPBasisGenerator
from mp_lib import dmps
import numpy as np
import gym


class DmpEnvWrapper(gym.Wrapper):
    def __init__(self,
                 env,
                 num_dof,
                 num_basis,
                 start_pos=None,
                 final_pos=None,
                 duration=1,
                 alpha_phase=2,
                 dt=0.01,
                 learn_goal=False,
                 post_traj_time=0.,
                 policy_type=None,
                 weights_scale=1.):
        super(DmpEnvWrapper, self).__init__(env)
        self.num_dof = num_dof
        self.num_basis = num_basis
        self.dim = num_dof * num_basis
        if learn_goal:
            self.dim += num_dof
        self.learn_goal = learn_goal
        self.duration = duration   # seconds
        time_steps = int(duration / dt)
        self.t = np.linspace(0, duration, time_steps)
        self.post_traj_steps = int(post_traj_time / dt)

        phase_generator = ExpDecayPhaseGenerator(alpha_phase=alpha_phase, duration=duration)
        basis_generator = DMPBasisGenerator(phase_generator, duration=duration, num_basis=self.num_basis)

        self.dmp = dmps.DMP(num_dof=num_dof,
                            basis_generator=basis_generator,
                            phase_generator=phase_generator,
                            num_time_steps=time_steps,
                            dt=dt
                            )

        self.dmp.dmp_start_pos = start_pos.reshape((1, num_dof))

        dmp_weights = np.zeros((num_basis, num_dof))
        if learn_goal:
            dmp_goal_pos = np.zeros(num_dof)
        else:
            dmp_goal_pos = final_pos

        self.dmp.set_weights(dmp_weights, dmp_goal_pos)
        self.weights_scale = weights_scale

        policy_class = get_policy_class(policy_type)
        self.policy = policy_class(env)

    def __call__(self, params, contexts=None):
        params = np.atleast_2d(params)
        rewards = []
        infos = []
        for p, c in zip(params, contexts):
            reward, info = self.rollout(p, c)
            rewards.append(reward)
            infos.append(info)

        return np.array(rewards), infos

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

        return goal_pos, weight_matrix * self.weights_scale

    def rollout(self, params, context=None, render=False):
        """ This function generates a trajectory based on a DMP and then does the usual loop over reset and step"""
        goal_pos, weight_matrix = self.goal_and_weights(params)
        self.dmp.set_weights(weight_matrix, goal_pos)
        trajectory, velocity = self.dmp.reference_trajectory(self.t)

        if self.post_traj_steps > 0:
            trajectory = np.vstack([trajectory, np.tile(trajectory[-1, :], [self.post_traj_steps, 1])])
            velocity = np.vstack([velocity, np.zeros(shape=(self.post_traj_steps, self.num_dof))])

        self._trajectory = trajectory
        self._velocity = velocity

        rews = []
        infos = []

        self.env.configure(context)
        self.env.reset()

        for t, pos_vel in enumerate(zip(trajectory, velocity)):
            ac = self.policy.get_action(pos_vel[0], pos_vel[1])
            obs, rew, done, info = self.env.step(ac)
            rews.append(rew)
            infos.append(info)
            if render:
                self.env.render(mode="human")
            if done:
                break

        reward = np.sum(rews)

        return reward, info
