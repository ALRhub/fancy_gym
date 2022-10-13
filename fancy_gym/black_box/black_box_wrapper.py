from typing import Tuple, Optional

import gym
import numpy as np
from gym import spaces
from mp_pytorch.mp.mp_interfaces import MPInterface

from fancy_gym.black_box.controller.base_controller import BaseController
from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper
from fancy_gym.utils.utils import get_numpy

import torch

class BlackBoxWrapper(gym.ObservationWrapper):

    def __init__(self,
                 env: RawInterfaceWrapper,
                 trajectory_generator: MPInterface,
                 tracking_controller: BaseController,
                 duration: float,
                 verbose: int = 1,
                 learn_sub_trajectories: bool = False,
                 replanning_schedule: Optional[callable] = None,
                 reward_aggregation: callable = np.sum
                 ):
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
        super().__init__(env)

        self.duration = duration
        self.learn_sub_trajectories = learn_sub_trajectories
        self.do_replanning = replanning_schedule is not None
        self.replanning_schedule = replanning_schedule or (lambda *x: False)
        self.current_traj_steps = 0

        # trajectory generation
        self.traj_gen = trajectory_generator

        self.tracking_controller = tracking_controller
        # self.time_steps = np.linspace(0, self.duration, self.traj_steps)
        # self.traj_gen.set_mp_times(self.time_steps)
        self.traj_gen.set_duration(self.duration, self.dt)
        # self.traj_gen.basis_gn.show_basis(plot=True)

        # reward computation
        self.reward_aggregation = reward_aggregation

        # spaces
        self.return_context_observation = not (learn_sub_trajectories or self.do_replanning)
        self.traj_gen_action_space = self._get_traj_gen_action_space()
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()

        # rendering
        self.render_kwargs = {}
        self.verbose = verbose

    def observation(self, observation):
        # return context space if we are
        if self.return_context_observation:
            observation = observation[self.env.context_mask]
        # cast dtype because metaworld returns incorrect that throws gym error
        return observation.astype(self.observation_space.dtype)

    def get_trajectory(self, action: np.ndarray) -> Tuple:
        clipped_params = np.clip(action, self.traj_gen_action_space.low, self.traj_gen_action_space.high)
        self.traj_gen.set_params(clipped_params)
        bc_time = np.array(0 if not self.do_replanning else self.current_traj_steps * self.dt)
        # TODO we could think about initializing with the previous desired value in order to have a smooth transition
        #  at least from the planning point of view.

        self.traj_gen.set_boundary_conditions(bc_time,
                                                self.current_pos,
                                                self.current_vel)
        duration = None if self.learn_sub_trajectories else self.duration
        self.traj_gen.set_duration(duration, self.dt)
        # traj_dict = self.traj_gen.get_trajs(get_pos=True, get_vel=True)
        trajectory = get_numpy(self.traj_gen.get_traj_pos())
        velocity = get_numpy(self.traj_gen.get_traj_vel())

        if self.do_replanning:
            # Remove first part of trajectory as this is already over
            trajectory = trajectory[self.current_traj_steps:]
            velocity = velocity[self.current_traj_steps:]

        return trajectory, velocity

    def _get_traj_gen_action_space(self):
        """This function can be used to set up an individual space for the parameters of the traj_gen."""
        min_action_bounds, max_action_bounds = self.traj_gen.get_params_bounds()
        action_space = gym.spaces.Box(low=min_action_bounds.numpy(), high=max_action_bounds.numpy(),
                                      dtype=self.env.action_space.dtype)
        return action_space

    def _get_action_space(self):
        """
        This function can be used to modify the action space for considering actions which are not learned via movement
        primitives. E.g. ball releasing time for the beer pong task. By default, it is the parameter space of the
        movement primitive.
        Only needs to be overwritten if the action space needs to be modified.
        """
        try:
            return self.traj_gen_action_space
        except AttributeError:
            return self._get_traj_gen_action_space()

    def _get_observation_space(self):
        if self.return_context_observation:
            mask = self.env.context_mask
            # return full observation
            min_obs_bound = self.env.observation_space.low[mask]
            max_obs_bound = self.env.observation_space.high[mask]
            return spaces.Box(low=min_obs_bound, high=max_obs_bound, dtype=self.env.observation_space.dtype)
        return self.env.observation_space

    def step(self, action: np.ndarray):
        """ This function generates a trajectory based on a MP and then does the usual loop over reset and step"""

        # TODO remove this part, right now only needed for beer pong
        mp_params, env_spec_params = self.env.episode_callback(action, self.traj_gen)
        trajectory, velocity = self.get_trajectory(mp_params)

        trajectory_length = len(trajectory)
        rewards = np.zeros(shape=(trajectory_length,))
        if self.verbose >= 2:
            actions = np.zeros(shape=(trajectory_length,) + self.env.action_space.shape)
            observations = np.zeros(shape=(trajectory_length,) + self.env.observation_space.shape,
                                    dtype=self.env.observation_space.dtype)

        infos = dict()
        done = False

        for t, (pos, vel) in enumerate(zip(trajectory, velocity)):
            step_action = self.tracking_controller.get_action(pos, vel, self.current_pos, self.current_vel)
            c_action = np.clip(step_action, self.env.action_space.low, self.env.action_space.high)
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
                self.env.render(**self.render_kwargs)

            if done or self.replanning_schedule(self.current_pos, self.current_vel, obs, c_action,
                                                t + 1 + self.current_traj_steps):
                break

        infos.update({k: v[:t+1] for k, v in infos.items()})
        self.current_traj_steps += t + 1

        if self.verbose >= 2:
            infos['positions'] = trajectory
            infos['velocities'] = velocity
            infos['step_actions'] = actions[:t + 1]
            infos['step_observations'] = observations[:t + 1]
            infos['step_rewards'] = rewards[:t + 1]

        infos['trajectory_length'] = t + 1
        trajectory_return = self.reward_aggregation(rewards[:t + 1])
        return self.observation(obs), trajectory_return, done, infos

    def render(self, **kwargs):
        """Only set render options here, such that they can be used during the rollout.
        This only needs to be called once"""
        self.render_kwargs = kwargs

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        self.current_traj_steps = 0
        return super(BlackBoxWrapper, self).reset()
