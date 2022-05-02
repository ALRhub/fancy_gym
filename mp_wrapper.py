from abc import ABC, abstractmethod
from typing import Union, Tuple

import gym
import numpy as np

from gym import spaces
from gym.envs.mujoco import MujocoEnv

from policies import get_policy_class, BaseController
from mp_pytorch.mp.mp_interfaces import MPInterface


class BaseMPWrapper(gym.Env, ABC):
    """
    Base class for movement primitive based gym.Wrapper implementations.

    Args:
        env: The (wrapped) environment this wrapper is applied on
        num_dof: Dimension of the action space of the wrapped env
        num_basis: Number of basis functions per dof
        duration: Length of the trajectory of the movement primitive in seconds
        post_traj_time: Time for which the last position of the trajectory is fed to the environment to continue
        simulation in seconds
        policy_type: Type or object defining the policy that is used to generate action based on the trajectory
        weight_scale: Scaling parameter for the actions given to this wrapper
        render_mode: Equivalent to gym render mode
    """

    def __init__(self,
                 env: MujocoEnv,
                 mp: MPInterface,
                 duration: float,
                 policy_type: Union[str, BaseController] = None,
                 render_mode: str = None,
                 verbose=2,
                 **mp_kwargs
                 ):
        super().__init__()

        assert env.dt is not None
        self.env = env
        self.dt = env.dt
        self.duration = duration
        self.traj_steps = int(duration / self.dt)
        self.post_traj_steps = self.env.spec.max_episode_steps - self.traj_steps

        # TODO: move to constructor, use policy factory instead what Fabian already coded
        if isinstance(policy_type, str):
            # pop policy kwargs here such that they are not passed to the initialize_mp method
            self.policy = get_policy_class(policy_type, self, **mp_kwargs.pop('policy_kwargs', {}))
        else:
            self.policy = policy_type

        self.mp = mp
        self.env = env
        self.verbose = verbose

        # rendering
        self.render_mode = render_mode
        self.render_kwargs = {}
        # self.time_steps = np.linspace(0, self.duration, self.traj_steps + 1)
        self.time_steps = np.linspace(0, self.duration, self.traj_steps)
        self.mp.set_mp_times(self.time_steps)

        # action_bounds = np.inf * np.ones((np.prod(self.mp.num_params)))
        min_action_bounds, max_action_bounds = mp.get_param_bounds()
        self.action_space = gym.spaces.Box(low=min_action_bounds.numpy(), high=max_action_bounds.numpy(),
                                           dtype=np.float32)

        self.active_obs = self.set_active_obs()
        self.observation_space = spaces.Box(low=self.env.observation_space.low[self.active_obs],
                                            high=self.env.observation_space.high[self.active_obs],
                                            dtype=self.env.observation_space.dtype)

    def get_trajectory(self, action: np.ndarray) -> Tuple:
        self.mp.set_params(action)
        self.mp.set_boundary_conditions(bc_time=self.time_steps[:1], bc_pos=self.current_pos, bc_vel=self.current_vel)
        traj_dict = self.mp.get_mp_trajs(get_pos = True, get_vel = True)
        trajectory_tensor, velocity_tensor = traj_dict['pos'], traj_dict['vel']

        trajectory = trajectory_tensor.numpy()
        velocity = velocity_tensor.numpy()

        if self.post_traj_steps > 0:
            trajectory = np.vstack([trajectory, np.tile(trajectory[-1, :], [self.post_traj_steps, 1])])
            velocity = np.vstack([velocity, np.zeros(shape=(self.post_traj_steps, self.mp.num_dof))])

        return trajectory, velocity

    @abstractmethod
    def set_active_obs(self):
        pass

    @property
    @abstractmethod
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        """
            Returns the current position of the action/control dimension.
            The dimensionality has to match the action/control dimension.
            This is not required when exclusively using velocity control,
            it should, however, be implemented regardless.
            E.g. The joint positions that are directly or indirectly controlled by the action.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        """
            Returns the current velocity of the action/control dimension.
            The dimensionality has to match the action/control dimension.
            This is not required when exclusively using position control,
            it should, however, be implemented regardless.
            E.g. The joint velocities that are directly or indirectly controlled by the action.
        """
        raise NotImplementedError()

    def _step_callback(self, t, action):
        pass

    def step(self, action: np.ndarray):
        """ This function generates a trajectory based on a MP and then does the usual loop over reset and step"""
        # TODO: Think about sequencing
        # TODO: Reward Function rather here?
        # agent to learn when to release the ball
        trajectory, velocity = self.get_trajectory(action)

        trajectory_length = len(trajectory)
        actions = np.zeros(shape=(trajectory_length,) + self.env.action_space.shape)
        if isinstance(self.env.observation_space, spaces.Dict):  # For goal environments
            observations = np.zeros(shape=(trajectory_length,) + self.env.observation_space["observation"].shape,
                                    dtype=self.env.observation_space.dtype)
        else:
            observations = np.zeros(shape=(trajectory_length,) + self.env.observation_space.shape,
                                    dtype=self.env.observation_space.dtype)
        rewards = np.zeros(shape=(trajectory_length,))
        trajectory_return = 0

        infos = dict()

        for t, pos_vel in enumerate(zip(trajectory, velocity)):
            ac = self.policy.get_action(pos_vel[0], pos_vel[1])
            callback_action = self._step_callback(t, action)
            if callback_action is not None:
                ac = np.concatenate((callback_action, ac))      # include callback action at first pos of vector
            actions[t, :] = np.clip(ac, self.env.action_space.low, self.env.action_space.high)
            obs, rewards[t], done, info = self.env.step(actions[t, :])
            observations[t, :] = obs["observation"] if isinstance(self.env.observation_space, spaces.Dict) else obs
            trajectory_return += rewards[t]
            for k, v in info.items():
                elems = infos.get(k, [None] * trajectory_length)
                elems[t] = v
                infos[k] = elems
            # infos['step_infos'].append(info)
            if self.render_mode:
                self.render(mode=self.render_mode, **self.render_kwargs)
            if done:
                break
        infos.update({k: v[:t + 1] for k, v in infos.items()})
        infos['trajectory'] = trajectory
        if self.verbose == 2:
            infos['step_actions'] = actions[:t + 1]
            infos['step_observations'] = observations[:t + 1]
            infos['step_rewards'] = rewards[:t + 1]
            infos['trajectory_length'] = t + 1
        done = True
        return self.get_observation_from_step(observations[t]), trajectory_return, done, infos

    def reset(self):
        return self.get_observation_from_step(self.env.reset())

    def render(self, mode='human', **kwargs):
        """Only set render options here, such that they can be used during the rollout.
        This only needs to be called once"""
        self.render_mode = mode
        self.render_kwargs = kwargs
        # self.env.render(mode=self.render_mode, **self.render_kwargs)
        self.env.render(mode=self.render_mode)

    def get_observation_from_step(self, observation: np.ndarray) -> np.ndarray:
        return observation[self.active_obs]

    def plot_trajs(self, des_trajs, des_vels):
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('TkAgg')
        pos_fig = plt.figure('positions')
        vel_fig = plt.figure('velocities')
        for i in range(des_trajs.shape[1]):
            plt.figure(pos_fig.number)
            plt.subplot(des_trajs.shape[1], 1, i + 1)
            plt.plot(np.ones(des_trajs.shape[0])*self.current_pos[i])
            plt.plot(des_trajs[:, i])

            plt.figure(vel_fig.number)
            plt.subplot(des_vels.shape[1], 1, i + 1)
            plt.plot(np.ones(des_trajs.shape[0])*self.current_vel[i])
            plt.plot(des_vels[:, i])
