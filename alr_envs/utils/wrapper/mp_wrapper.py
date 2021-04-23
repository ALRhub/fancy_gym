from abc import ABC, abstractmethod
from collections import defaultdict

import gym
import numpy as np

from alr_envs.utils.policies import get_policy_class


class MPWrapper(gym.Wrapper, ABC):

    def __init__(self,
                 env: gym.Env,
                 num_dof: int,
                 duration: int = 1,
                 dt: float = None,
                 post_traj_time: float = 0.,
                 policy_type: str = None,
                 weights_scale: float = 1.,
                 **mp_kwargs
                 ):
        super().__init__(env)

        # self.num_dof = num_dof
        # self.num_basis = num_basis
        # self.duration = duration  # seconds

        # dt = env.dt if hasattr(env, "dt") else dt
        assert dt is not None  # this should never happen as MPWrapper is a base class
        self.post_traj_steps = int(post_traj_time / dt)

        self.mp = self.initialize_mp(num_dof, duration, dt, **mp_kwargs)
        self.weights_scale = weights_scale

        policy_class = get_policy_class(policy_type)
        self.policy = policy_class(env)

        # rendering
        self.render_mode = None
        self.render_kwargs = None

    # TODO: not yet final
    def __call__(self, params, contexts=None):
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

    def configure(self, context):
        self.env.configure(context)

    def step(self, action: np.ndarray):
        """ This function generates a trajectory based on a DMP and then does the usual loop over reset and step"""
        trajectory, velocity = self.mp_rollout(action)

        if self.post_traj_steps > 0:
            trajectory = np.vstack([trajectory, np.tile(trajectory[-1, :], [self.post_traj_steps, 1])])
            velocity = np.vstack([velocity, np.zeros(shape=(self.post_traj_steps, self.dmp.num_dimensions))])

        # self._trajectory = trajectory
        # self._velocity = velocity

        rewards = 0
        # infos = defaultdict(list)

        # TODO: @Max Why do we need this configure, states should be part of the model
        # TODO: Ask Onur if the context distribution needs to be outside the environment
        # self.env.configure(context)
        obs = self.env.reset()
        info = {}

        for t, pos_vel in enumerate(zip(trajectory, velocity)):
            ac = self.policy.get_action(pos_vel[0], pos_vel[1])
            obs, rew, done, info = self.env.step(ac)
            rewards += rew
            # TODO return all dicts?
            # [infos[k].append(v) for k, v in info.items()]
            if self.render_mode:
                self.env.render(mode=self.render_mode, **self.render_kwargs)
            if done:
                break

        done = True
        return obs, rewards, done, info

    def render(self, mode='human', **kwargs):
        """Only set render options here, such that they can be used during the rollout.
        This only needs to be called once"""
        self.render_mode = mode
        self.render_kwargs = kwargs

    # def __call__(self, actions):
    #     return self.step(actions)
        # params = np.atleast_2d(params)
        # rewards = []
        # infos = []
        # for p, c in zip(params, contexts):
        #     reward, info = self.rollout(p, c)
        #     rewards.append(reward)
        #     infos.append(info)
        #
        # return np.array(rewards), infos

    @abstractmethod
    def mp_rollout(self, action):
        """
        Generate trajectory and velocity based on the MP
        Returns:
            trajectory/positions, velocity
        """
        raise NotImplementedError()

    @abstractmethod
    def initialize_mp(self, num_dof: int, duration: int, dt: float, **kwargs):
        """
        Create respective instance of MP
        Returns:
             MP instance
        """

        raise NotImplementedError
