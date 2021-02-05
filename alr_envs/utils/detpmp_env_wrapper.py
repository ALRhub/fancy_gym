from mp_lib import det_promp
import numpy as np
import gym


class DetPMPEnvWrapperBase(gym.Wrapper):
    def __init__(self,
                 env,
                 num_dof,
                 num_basis,
                 width,
                 start_pos=None,
                 duration=1,
                 dt=0.01,
                 post_traj_time=0.,
                 policy=None,
                 weights_scale=1):
        super(DetPMPEnvWrapperBase, self).__init__(env)
        self.num_dof = num_dof
        self.num_basis = num_basis
        self.dim = num_dof * num_basis
        self.pmp = det_promp.DeterministicProMP(n_basis=num_basis, width=width, off=0.01)
        weights = np.zeros(shape=(num_basis, num_dof))
        self.pmp.set_weights(duration, weights)
        self.weights_scale = weights_scale

        self.duration = duration
        self.dt = dt
        self.post_traj_steps = int(post_traj_time / dt)

        self.start_pos = start_pos

        self.policy = policy

    def __call__(self, params):
        params = np.atleast_2d(params)
        observations = []
        rewards = []
        dones = []
        infos = []
        for p in params:
            observation, reward, done, info = self.rollout(p)
            observations.append(observation)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        return np.array(rewards), infos

    def rollout(self, params, render=False):
        """ This function generates a trajectory based on a DMP and then does the usual loop over reset and step"""
        raise NotImplementedError


class DetPMPEnvWrapperPD(DetPMPEnvWrapperBase):
    """
    Wrapper for gym environments which creates a trajectory in joint velocity space
    """
    def rollout(self, params, render=False):
        params = np.reshape(params, newshape=(self.num_basis, self.num_dof)) * self.weights_scale
        self.pmp.set_weights(self.duration, params)
        t, des_pos, des_vel, des_acc = self.pmp.compute_trajectory(1/self.dt, 1.)
        des_pos += self.start_pos[None, :]

        if self.post_traj_steps > 0:
            des_pos = np.vstack([des_pos, np.tile(des_pos[-1, :], [self.post_traj_steps, 1])])
            des_vel = np.vstack([des_vel, np.zeros(shape=(self.post_traj_steps, self.num_dof))])

        self._trajectory = des_pos

        rews = []
        infos = []

        self.env.reset()

        for t, pos_vel in enumerate(zip(des_pos, des_vel)):
            ac = self.policy.get_action(self.env, pos_vel[0], pos_vel[1])
            obs, rew, done, info = self.env.step(ac)
            rews.append(rew)
            infos.append(info)
            if render:
                self.env.render(mode="human")
            if done:
                break

        reward = np.sum(rews)

        return obs, reward, done, info
