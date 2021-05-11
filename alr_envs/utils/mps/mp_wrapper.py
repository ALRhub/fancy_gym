from abc import ABC, abstractmethod
from typing import Union

import gym
import numpy as np

from alr_envs.utils.mps.alr_env import AlrEnv
from alr_envs.utils.policies import get_policy_class, BaseController


class MPWrapper(gym.Wrapper, ABC):
    """
    Base class for movement primitive based gym.Wrapper implementations.

    :param env: The (wrapped) environment this wrapper is applied on
    :param num_dof: Dimension of the action space of the wrapped env
    :param duration: Number of timesteps in the trajectory of the movement primitive
    :param post_traj_time: Time for which the last position of the trajectory is fed to the environment to continue
    simulation
    :param policy_type: Type or object defining the policy that is used to generate action based on the trajectory
    :param weight_scale: Scaling parameter for the actions given to this wrapper
    :param render_mode: Equivalent to gym render mode
    """
    def __init__(self,
                 env: AlrEnv,
                 num_dof: int,
                 duration: int = 1,
                 post_traj_time: float = 0.,
                 policy_type: Union[str, BaseController] = None,
                 weights_scale: float = 1.,
                 render_mode: str = None,
                 **mp_kwargs
                 ):
        super().__init__(env)

        # adjust observation space to reduce version
        obs_sp = self.env.observation_space
        self.observation_space = gym.spaces.Box(low=obs_sp.low[self.env.active_obs],
                                                high=obs_sp.high[self.env.active_obs],
                                                dtype=obs_sp.dtype)

        self.post_traj_steps = int(post_traj_time / env.dt)

        self.mp = self.initialize_mp(num_dof=num_dof, duration=duration, **mp_kwargs)
        self.weights_scale = weights_scale

        if type(policy_type) is str:
            self.policy = get_policy_class(policy_type, env, mp_kwargs)
        else:
            self.policy = policy_type

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
            velocity = np.vstack([velocity, np.zeros(shape=(self.post_traj_steps, self.mp.num_dimensions))])

        trajectory_length = len(trajectory)
        actions = np.zeros(shape=(trajectory_length, self.mp.num_dimensions))
        observations= np.zeros(shape=(trajectory_length,) + self.env.observation_space.shape)
        rewards = np.zeros(shape=(trajectory_length,))
        trajectory_return = 0
        infos = dict(step_infos =[])
        for t, pos_vel in enumerate(zip(trajectory, velocity)):
            actions[t,:] = self.policy.get_action(pos_vel[0], pos_vel[1])
            observations[t,:], rewards[t], done, info = self.env.step(actions[t,:])
            trajectory_return += rewards[t]
            infos['step_infos'].append(info)
            if self.render_mode:
                self.env.render(mode=self.render_mode, **self.render_kwargs)
            if done:
                break

        infos['step_actions'] = actions[:t+1]
        infos['step_observations'] = observations[:t+1]
        infos['step_rewards'] = rewards[:t+1]
        infos['trajectory_length'] = t+1
        done = True
        return observations[t][self.env.active_obs], trajectory_return, done, infos

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
    def initialize_mp(self, num_dof: int, duration: float, **kwargs):
        """
        Create respective instance of MP
        Returns:
             MP instance
        """

        raise NotImplementedError
