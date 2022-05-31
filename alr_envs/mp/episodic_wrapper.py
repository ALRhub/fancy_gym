from abc import ABC, abstractmethod
from typing import Union, Tuple

import gym
import numpy as np
from gym import spaces
from mp_pytorch.mp.mp_interfaces import MPInterface

from alr_envs.mp.controllers.base_controller import BaseController


class EpisodicWrapper(gym.Env, ABC):
    """
    Base class for movement primitive based gym.Wrapper implementations.

    Args:
        env: The (wrapped) environment this wrapper is applied on
        num_dof: Dimension of the action space of the wrapped env
        num_basis: Number of basis functions per dof
        duration: Length of the trajectory of the movement primitive in seconds
        controller: Type or object defining the policy that is used to generate action based on the trajectory
        weight_scale: Scaling parameter for the actions given to this wrapper
        render_mode: Equivalent to gym render mode
    """
    def __init__(self,
                 env: gym.Env,
                 mp: MPInterface,
                 controller: BaseController,
                 duration: float,
                 render_mode: str = None,
                 verbose: int = 1,
                 weight_scale: float = 1):
        super().__init__()

        self.env = env
        try:
            self.dt = env.dt
        except AttributeError:
            raise AttributeError("step based environment needs to have a function 'dt' ")
        self.duration = duration
        self.traj_steps = int(duration / self.dt)
        self.post_traj_steps = self.env.spec.max_episode_steps - self.traj_steps

        self.controller = controller
        self.mp = mp
        self.env = env
        self.verbose = verbose
        self.weight_scale = weight_scale

        # rendering
        self.render_mode = render_mode
        self.render_kwargs = {}
        self.time_steps = np.linspace(0, self.duration, self.traj_steps)
        self.mp.set_mp_times(self.time_steps)
        # action_bounds = np.inf * np.ones((np.prod(self.mp.num_params)))
        self.mp_action_space = self.set_mp_action_space()

        self.action_space = self.set_action_space()
        self.active_obs = self.set_active_obs()
        self.observation_space = spaces.Box(low=self.env.observation_space.low[self.active_obs],
                                            high=self.env.observation_space.high[self.active_obs],
                                            dtype=self.env.observation_space.dtype)

    def get_trajectory(self, action: np.ndarray) -> Tuple:
        # TODO: this follows the implementation of the mp_pytorch library which includes the parameters tau and delay at
        #  the beginning of the array.
        ignore_indices = int(self.mp.learn_tau) + int(self.mp.learn_delay)
        scaled_mp_params = action.copy()
        scaled_mp_params[ignore_indices:] *= self.weight_scale
        self.mp.set_params(np.clip(scaled_mp_params, self.mp_action_space.low, self.mp_action_space.high))
        self.mp.set_boundary_conditions(bc_time=self.time_steps[:1], bc_pos=self.current_pos, bc_vel=self.current_vel)
        traj_dict = self.mp.get_mp_trajs(get_pos = True, get_vel = True)
        trajectory_tensor, velocity_tensor = traj_dict['pos'], traj_dict['vel']

        trajectory = trajectory_tensor.numpy()
        velocity = velocity_tensor.numpy()

        if self.post_traj_steps > 0:
            trajectory = np.vstack([trajectory, np.tile(trajectory[-1, :], [self.post_traj_steps, 1])])
            velocity = np.vstack([velocity, np.zeros(shape=(self.post_traj_steps, self.mp.num_dof))])

        return trajectory, velocity

    def set_mp_action_space(self):
        """This function can be used to set up an individual space for the parameters of the mp."""
        min_action_bounds, max_action_bounds = self.mp.get_param_bounds()
        mp_action_space = gym.spaces.Box(low=min_action_bounds.numpy(), high=max_action_bounds.numpy(),
                                              dtype=np.float32)
        return mp_action_space

    def set_action_space(self):
        """
        This function can be used to modify the action space for considering actions which are not learned via motion
        primitives. E.g. ball releasing time for the beer pong task. By default, it is the parameter space of the
        motion primitive.
        Only needs to be overwritten if the action space needs to be modified.
        """
        return self.mp_action_space

    def _episode_callback(self, action: np.ndarray) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
        """
        Used to extract the parameters for the motion primitive and other parameters from an action array which might
        include other actions like ball releasing time for the beer pong environment.
        This only needs to be overwritten if the action space is modified.
        Args:
            action: a vector instance of the whole action space, includes mp parameters and additional parameters if
            specified, else only mp parameters

        Returns:
            Tuple: mp_arguments and other arguments
        """
        return action, None

    def _step_callback(self, t: int, env_spec_params: Union[np.ndarray, None], step_action: np.ndarray) -> Union[np.ndarray]:
        """
        This function can be used to modify the step_action with additional parameters e.g. releasing the ball in the
        Beerpong env. The parameters used should not be part of the motion primitive parameters.
        Returns step_action by default, can be overwritten in individual mp_wrappers.
        Args:
            t: the current time step of the episode
            env_spec_params: the environment specific parameter, as defined in fucntion _episode_callback
            (e.g. ball release time in Beer Pong)
            step_action: the current step-based action

        Returns:
            modified step action
        """
        return step_action

    @abstractmethod
    def set_active_obs(self) -> np.ndarray:
        """
        This function defines the contexts. The contexts are defined as specific observations.
        Returns:
            boolearn array representing the indices of the observations

        """
        return np.ones(self.env.observation_space.shape[0], dtype=bool)

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

    def step(self, action: np.ndarray):
        """ This function generates a trajectory based on a MP and then does the usual loop over reset and step"""
        # TODO: Think about sequencing
        # TODO: Reward Function rather here?
        # agent to learn when to release the ball
        mp_params, env_spec_params = self._episode_callback(action)
        trajectory, velocity = self.get_trajectory(mp_params)

        trajectory_length = len(trajectory)
        if self.verbose >=2 :
            actions = np.zeros(shape=(trajectory_length,) + self.env.action_space.shape)
            observations = np.zeros(shape=(trajectory_length,) + self.env.observation_space.shape,
                                        dtype=self.env.observation_space.dtype)
            rewards = np.zeros(shape=(trajectory_length,))
        trajectory_return = 0

        infos = dict()

        for t, pos_vel in enumerate(zip(trajectory, velocity)):
            step_action = self.controller.get_action(pos_vel[0], pos_vel[1], self.current_pos, self.current_vel)
            step_action = self._step_callback(t, env_spec_params, step_action)   # include possible callback info
            c_action = np.clip(step_action, self.env.action_space.low, self.env.action_space.high)
            # print('step/clipped action ratio: ', step_action/c_action)
            obs, c_reward, done, info = self.env.step(c_action)
            if self.verbose >= 2:
                actions[t, :] = c_action
                rewards[t] = c_reward
                observations[t, :] = obs
            trajectory_return += c_reward
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
        if self.verbose >= 2:
            infos['trajectory'] = trajectory
            infos['step_actions'] = actions[:t + 1]
            infos['step_observations'] = observations[:t + 1]
            infos['step_rewards'] = rewards[:t + 1]
        infos['trajectory_length'] = t + 1
        done = True
        return self.get_observation_from_step(obs), trajectory_return, done, infos

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

    def seed(self, seed=None):
        self.env.seed(seed)

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
