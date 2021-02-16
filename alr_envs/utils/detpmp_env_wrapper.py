from alr_envs.utils.policies import get_policy_class
from mp_lib import det_promp
import numpy as np
import gym


class DetPMPEnvWrapper(gym.Wrapper):
    def __init__(self,
                 env,
                 num_dof,
                 num_basis,
                 width,
                 start_pos=None,
                 duration=1,
                 dt=0.01,
                 post_traj_time=0.,
                 policy_type=None,
                 weights_scale=1,
                 zero_start=False,
                 zero_goal=False,
                 ):
        super(DetPMPEnvWrapper, self).__init__(env)
        self.num_dof = num_dof
        self.num_basis = num_basis
        self.dim = num_dof * num_basis
        self.pmp = det_promp.DeterministicProMP(n_basis=num_basis, n_dof=num_dof, width=width, off=0.01,
                                                zero_start=zero_start, zero_goal=zero_goal)
        weights = np.zeros(shape=(num_basis, num_dof))
        self.pmp.set_weights(duration, weights)
        self.weights_scale = weights_scale

        self.duration = duration
        self.dt = dt
        self.post_traj_steps = int(post_traj_time / dt)

        self.start_pos = start_pos
        self.zero_centered = zero_start

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

    def rollout(self, params, context=None, render=False):
        """ This function generates a trajectory based on a DMP and then does the usual loop over reset and step"""
        params = np.reshape(params, newshape=(self.num_basis, self.num_dof)) * self.weights_scale
        self.pmp.set_weights(self.duration, params)
        t, des_pos, des_vel, des_acc = self.pmp.compute_trajectory(1 / self.dt, 1.)
        if self.zero_centered:
            des_pos += self.start_pos[None, :]

        if self.post_traj_steps > 0:
            des_pos = np.vstack([des_pos, np.tile(des_pos[-1, :], [self.post_traj_steps, 1])])
            des_vel = np.vstack([des_vel, np.zeros(shape=(self.post_traj_steps, self.num_dof))])

        self._trajectory = des_pos
        self._velocity = des_vel

        rews = []
        infos = []

        self.env.configure(context)
        self.env.reset()

        for t, pos_vel in enumerate(zip(des_pos, des_vel)):
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

