from abc import ABC
from typing import Tuple, Union

import gym
import numpy as np
from gym import spaces
from mp_pytorch.mp.mp_interfaces import MPInterface

from alr_envs.mp.controllers.base_controller import BaseController
from alr_envs.mp.raw_interface_wrapper import RawInterfaceWrapper
from alr_envs.utils.utils import get_numpy


class BlackBoxWrapper(gym.ObservationWrapper, ABC):

    def __init__(self,
                 env: RawInterfaceWrapper,
                 trajectory_generator: MPInterface, tracking_controller: BaseController,
                 duration: float, verbose: int = 1, learn_sub_trajectories: bool = False,
                 replanning_schedule: Union[None, callable] = None,
                 reward_aggregation: callable = np.sum):
        """
        gym.Wrapper for leveraging a black box approach with a trajectory generator.

        Args:
            env: The (wrapped) environment this wrapper is applied on
            trajectory_generator: Generates the full or partial trajectory
            tracking_controller: Translates the desired trajectory to raw action sequences
            duration: Length of the trajectory of the movement primitive in seconds
            verbose: level of detail for returned values in info dict.
            learn_sub_trajectories: Transforms full episode learning into learning sub-trajectories, similar to
                step-based learning
            replanning_schedule: callable that receives
            reward_aggregation: function that takes the np.ndarray of step rewards as input and returns the trajectory
                reward, default summation over all values.
        """
        super().__init__()

        self.env = env
        self.duration = duration
        self.learn_sub_trajectories = learn_sub_trajectories
        self.replanning_schedule = replanning_schedule
        self.current_traj_steps = 0

        # trajectory generation
        self.traj_gen = trajectory_generator
        self.tracking_controller = tracking_controller
        # self.time_steps = np.linspace(0, self.duration, self.traj_steps)
        # self.traj_gen.set_mp_times(self.time_steps)
        self.traj_gen.set_duration(np.array([self.duration]), np.array([self.dt]))

        # reward computation
        self.reward_aggregation = reward_aggregation

        # spaces
        self.return_context_observation = not (self.learn_sub_trajectories or replanning_schedule)
        self.traj_gen_action_space = self.get_traj_gen_action_space()
        self.action_space = self.get_action_space()
        self.observation_space = spaces.Box(low=self.env.observation_space.low[self.env.context_mask],
                                            high=self.env.observation_space.high[self.env.context_mask],
                                            dtype=self.env.observation_space.dtype)

        # rendering
        self.render_kwargs = {}
        self.verbose = verbose

    def observation(self, observation):
        # return context space if we are
        return observation[self.env.context_mask] if self.return_context_observation else observation

    def get_trajectory(self, action: np.ndarray) -> Tuple:
        clipped_params = np.clip(action, self.traj_gen_action_space.low, self.traj_gen_action_space.high)
        self.traj_gen.set_params(clipped_params)
        # TODO: Bruce said DMP, ProMP, ProDMP can have 0 bc_time for sequencing
        # TODO Check with Bruce for replanning
        self.traj_gen.set_boundary_conditions(
            bc_time=np.zeros((1,)) if not self.replanning_schedule else self.current_traj_steps * self.dt,
            bc_pos=self.current_pos, bc_vel=self.current_vel)
        # TODO: is this correct for replanning? Do we need to adjust anything here?
        self.traj_gen.set_duration(None if self.learn_sub_trajectories else self.duration, np.array([self.dt]))
        traj_dict = self.traj_gen.get_trajs(get_pos=True, get_vel=True)
        trajectory_tensor, velocity_tensor = traj_dict['pos'], traj_dict['vel']

        return get_numpy(trajectory_tensor), get_numpy(velocity_tensor)

    def get_traj_gen_action_space(self):
        """This function can be used to set up an individual space for the parameters of the traj_gen."""
        min_action_bounds, max_action_bounds = self.traj_gen.get_param_bounds()
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
            return self.traj_gen_action_space
        except AttributeError:
            return self.get_traj_gen_action_space()

    def step(self, action: np.ndarray):
        """ This function generates a trajectory based on a MP and then does the usual loop over reset and step"""

        # agent to learn when to release the ball
        mp_params, env_spec_params = self._episode_callback(action)
        trajectory, velocity = self.get_trajectory(mp_params)

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

            if self.render_kwargs:
                self.render(**self.render_kwargs)

            if done:
                break

            if self.replanning_schedule and self.replanning_schedule(self.current_pos, self.current_vel, obs, c_action,
                                                                     t + 1 + self.current_traj_steps):
                break

        infos.update({k: v[:t + 1] for k, v in infos.items()})
        self.current_traj_steps += t + 1

        if self.verbose >= 2:
            infos['trajectory'] = trajectory
            infos['step_actions'] = actions[:t + 1]
            infos['step_observations'] = observations[:t + 1]
            infos['step_rewards'] = rewards[:t + 1]

        infos['trajectory_length'] = t + 1
        trajectory_return = self.reward_aggregation(rewards[:t + 1])
        return obs, trajectory_return, done, infos

    def render(self, **kwargs):
        """Only set render options here, such that they can be used during the rollout.
        This only needs to be called once"""
        self.render_kwargs = kwargs
        # self.env.render(mode=self.render_mode, **self.render_kwargs)
        self.env.render(**kwargs)

    def reset(self, **kwargs):
        self.current_traj_steps = 0
        super(BlackBoxWrapper, self).reset(**kwargs)

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
