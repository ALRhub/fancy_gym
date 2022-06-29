from abc import ABC
from typing import Tuple

import gym
import numpy as np
from gym import spaces
from mp_pytorch.mp.mp_interfaces import MPInterface

from alr_envs.mp.controllers.base_controller import BaseController
from alr_envs.mp.raw_interface_wrapper import RawInterfaceWrapper


class BlackBoxWrapper(gym.ObservationWrapper, ABC):

    def __init__(self,
                 env: RawInterfaceWrapper,
                 trajectory_generator: MPInterface, tracking_controller: BaseController,
                 duration: float, verbose: int = 1, sequencing=True, reward_aggregation: callable = np.sum):
        """
        gym.Wrapper for leveraging a black box approach with a trajectory generator.

        Args:
            env: The (wrapped) environment this wrapper is applied on
            trajectory_generator: Generates the full or partial trajectory
            tracking_controller: Translates the desired trajectory to raw action sequences
            duration: Length of the trajectory of the movement primitive in seconds
            verbose: level of detail for returned values in info dict.
            reward_aggregation: function that takes the np.ndarray of step rewards as input and returns the trajectory
                reward, default summation over all values.
        """
        super().__init__()

        self.env = env
        self.duration = duration
        self.traj_steps = int(duration / self.dt)
        self.post_traj_steps = self.env.spec.max_episode_steps - self.traj_steps
        # duration = self.env.max_episode_steps * self.dt

        # trajectory generation
        self.trajectory_generator = trajectory_generator
        self.tracking_controller = tracking_controller
        # self.weight_scale = weight_scale
        self.time_steps = np.linspace(0, self.duration, self.traj_steps)
        self.trajectory_generator.set_mp_times(self.time_steps)
        # self.trajectory_generator.set_mp_duration(self.time_steps, dt)
        # action_bounds = np.inf * np.ones((np.prod(self.trajectory_generator.num_params)))
        self.reward_aggregation = reward_aggregation

        # spaces
        self.mp_action_space = self.get_mp_action_space()
        self.action_space = self.get_action_space()
        self.observation_space = spaces.Box(low=self.env.observation_space.low[self.env.context_mask],
                                            high=self.env.observation_space.high[self.env.context_mask],
                                            dtype=self.env.observation_space.dtype)

        # rendering
        self.render_mode = None
        self.render_kwargs = {}

        self.verbose = verbose

    @property
    def dt(self):
        return self.env.dt

    def observation(self, observation):
        return observation[self.env.context_mask]

    def get_trajectory(self, action: np.ndarray) -> Tuple:
        # TODO: this follows the implementation of the mp_pytorch library which includes the parameters tau and delay at
        #  the beginning of the array.
        # ignore_indices = int(self.trajectory_generator.learn_tau) + int(self.trajectory_generator.learn_delay)
        # scaled_mp_params = action.copy()
        # scaled_mp_params[ignore_indices:] *= self.weight_scale

        clipped_params = np.clip(action, self.mp_action_space.low, self.mp_action_space.high)
        self.trajectory_generator.set_params(clipped_params)
        self.trajectory_generator.set_boundary_conditions(bc_time=self.time_steps[:1], bc_pos=self.current_pos,
                                                          bc_vel=self.current_vel)
        traj_dict = self.trajectory_generator.get_mp_trajs(get_pos=True, get_vel=True)
        trajectory_tensor, velocity_tensor = traj_dict['pos'], traj_dict['vel']

        trajectory = trajectory_tensor.numpy()
        velocity = velocity_tensor.numpy()

        # TODO: Do we need this or does mp_pytorch have this?
        if self.post_traj_steps > 0:
            trajectory = np.vstack([trajectory, np.tile(trajectory[-1, :], [self.post_traj_steps, 1])])
            velocity = np.vstack([velocity, np.zeros(shape=(self.post_traj_steps, self.trajectory_generator.num_dof))])

        return trajectory, velocity

    def get_mp_action_space(self):
        """This function can be used to set up an individual space for the parameters of the trajectory_generator."""
        min_action_bounds, max_action_bounds = self.trajectory_generator.get_param_bounds()
        mp_action_space = gym.spaces.Box(low=min_action_bounds.numpy(), high=max_action_bounds.numpy(),
                                         dtype=np.float32)
        return mp_action_space

    def get_action_space(self):
        """
        This function can be used to modify the action space for considering actions which are not learned via motion
        primitives. E.g. ball releasing time for the beer pong task. By default, it is the parameter space of the
        motion primitive.
        Only needs to be overwritten if the action space needs to be modified.
        """
        try:
            return self.mp_action_space
        except AttributeError:
            return self.get_mp_action_space()

    def step(self, action: np.ndarray):
        """ This function generates a trajectory based on a MP and then does the usual loop over reset and step"""
        # TODO: Think about sequencing
        # TODO: Reward Function rather here?
        # agent to learn when to release the ball
        mp_params, env_spec_params = self._episode_callback(action)
        trajectory, velocity = self.get_trajectory(mp_params)

        # TODO
        # self.time_steps = np.linspace(0, learned_duration, self.traj_steps)
        # self.trajectory_generator.set_mp_times(self.time_steps)

        trajectory_length = len(trajectory)
        rewards = np.zeros(shape=(trajectory_length,))
        if self.verbose >= 2:
            actions = np.zeros(shape=(trajectory_length,) + self.env.action_space.shape)
            observations = np.zeros(shape=(trajectory_length,) + self.env.observation_space.shape,
                                    dtype=self.env.observation_space.dtype)

        infos = dict()
        done = False

        for t, pos_vel in enumerate(zip(trajectory, velocity)):
            step_action = self.tracking_controller.get_action(pos_vel[0], pos_vel[1], self.current_pos,
                                                              self.current_vel)
            step_action = self._step_callback(t, env_spec_params, step_action)  # include possible callback info
            c_action = np.clip(step_action, self.env.action_space.low, self.env.action_space.high)
            # print('step/clipped action ratio: ', step_action/c_action)
            obs, c_reward, done, info = self.env.step(c_action)
            rewards[t] = c_reward

            if self.verbose >= 2:
                actions[t, :] = c_action
                observations[t, :] = obs

            for k, v in info.items():
                elems = infos.get(k, [None] * trajectory_length)
                elems[t] = v
                infos[k] = elems

            if self.render_mode is not None:
                self.render(mode=self.render_mode, **self.render_kwargs)

            if done or self.env.do_replanning(self.env.current_pos, self.env.current_vel, obs, c_action, t):
                break

        infos.update({k: v[:t + 1] for k, v in infos.items()})

        if self.verbose >= 2:
            infos['trajectory'] = trajectory
            infos['step_actions'] = actions[:t + 1]
            infos['step_observations'] = observations[:t + 1]
            infos['step_rewards'] = rewards[:t + 1]

        infos['trajectory_length'] = t + 1
        trajectory_return = self.reward_aggregation(rewards[:t + 1])
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
            plt.plot(np.ones(des_trajs.shape[0]) * self.current_pos[i])
            plt.plot(des_trajs[:, i])

            plt.figure(vel_fig.number)
            plt.subplot(des_vels.shape[1], 1, i + 1)
            plt.plot(np.ones(des_trajs.shape[0]) * self.current_vel[i])
            plt.plot(des_vels[:, i])
