from abc import ABC, abstractmethod

import gym
import numpy as np

from alr_envs.utils.mps.mp_environments import AlrEnv
from alr_envs.utils.policies import get_policy_class


class MPWrapper(gym.Wrapper, ABC):

    def __init__(self, env: AlrEnv, num_dof: int, dt: float, duration: float = 1, post_traj_time: float = 0.,
                 policy_type: str = None, weights_scale: float = 1., render_mode: str = None, **mp_kwargs):
        super().__init__(env)

        # adjust observation space to reduce version
        obs_sp = self.env.observation_space
        self.observation_space = gym.spaces.Box(low=obs_sp.low[self.env.active_obs],
                                                high=obs_sp.high[self.env.active_obs],
                                                dtype=obs_sp.dtype)

        assert dt is not None  # this should never happen as MPWrapper is a base class
        self.post_traj_steps = int(post_traj_time / dt)

        self.mp = self.initialize_mp(num_dof, duration, dt, **mp_kwargs)
        self.weights_scale = weights_scale

        policy_class = get_policy_class(policy_type)
        self.policy = policy_class(env)

        # rendering
        self.render_mode = render_mode
        self.render_kwargs = {}

    # TODO: @Max I think this should not be in this class, this functionality should be part of your sampler.
    def __call__(self, params, contexts=None):
        """
        Can be used to provide a batch of parameter sets
        """
        params = np.atleast_2d(params)
        obs = []
        rewards = []
        dones = []
        infos = []
        # for p, c in zip(params, contexts):
        for p in params:
            # self.configure(c)
            ob, reward, done, info = self.step(p)
            obs.append(ob)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        return obs, np.array(rewards), dones, infos

    def reset(self):
        return self.env.reset()[self.env.active_obs]

    def step(self, action: np.ndarray):
        """ This function generates a trajectory based on a DMP and then does the usual loop over reset and step"""
        trajectory, velocity = self.mp_rollout(action)

        if self.post_traj_steps > 0:
            trajectory = np.vstack([trajectory, np.tile(trajectory[-1, :], [self.post_traj_steps, 1])])
            velocity = np.vstack([velocity, np.zeros(shape=(self.post_traj_steps, self.mp.n_dof))])

        # self._trajectory = trajectory
        # self._velocity = velocity

        rewards = 0
        info = {}
        # create random obs as the reset function is called externally
        obs = self.env.observation_space.sample()

        for t, pos_vel in enumerate(zip(trajectory, velocity)):
            ac = self.policy.get_action(pos_vel[0], pos_vel[1])
            ac = np.clip(ac, self.env.action_space.low, self.env.action_space.high)
            obs, rew, done, info = self.env.step(ac)
            rewards += rew
            # TODO return all dicts?
            # [infos[k].append(v) for k, v in info.items()]
            if self.render_mode:
                self.env.render(mode=self.render_mode, **self.render_kwargs)
            if done:
                break

        done = True
        return obs[self.env.active_obs], rewards, done, info

    def render(self, mode='human', **kwargs):
        """Only set render options here, such that they can be used during the rollout.
        This only needs to be called once"""
        self.render_mode = mode
        self.render_kwargs = kwargs

    @abstractmethod
    def mp_rollout(self, action):
        """
        Generate trajectory and velocity based on the MP
        Returns:
            trajectory/positions, velocity
        """
        raise NotImplementedError()

    @abstractmethod
    def initialize_mp(self, num_dof: int, duration: float, dt: float, **kwargs):
        """
        Create respective instance of MP
        Returns:
             MP instance
        """

        raise NotImplementedError
