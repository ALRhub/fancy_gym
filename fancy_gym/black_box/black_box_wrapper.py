from typing import Tuple, Optional, Callable, Dict, Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ObsType
from mp_pytorch.mp.mp_interfaces import MPInterface

from fancy_gym.black_box.controller.base_controller import BaseController
from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper
from fancy_gym.utils.utils import get_numpy


class BlackBoxWrapper(gym.ObservationWrapper):

    def __init__(self,
                 env: RawInterfaceWrapper,
                 trajectory_generator: MPInterface,
                 tracking_controller: BaseController,
                 duration: float,
                 verbose: int = 1,
                 learn_sub_trajectories: bool = False,
                 replanning_schedule: Optional[
                     Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int], bool]] = None,
                 reward_aggregation: Callable[[np.ndarray], float] = np.sum,
                 max_planning_times: int = np.inf,
                 condition_on_desired: bool = False
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

        # check
        self.tau_bound = [-np.inf, np.inf]
        self.delay_bound = [-np.inf, np.inf]
        if hasattr(self.traj_gen.phase_gn, "tau_bound"):
            self.tau_bound = self.traj_gen.phase_gn.tau_bound
        if hasattr(self.traj_gen.phase_gn, "delay_bound"):
            self.delay_bound = self.traj_gen.phase_gn.delay_bound

        # reward computation
        self.reward_aggregation = reward_aggregation

        # spaces
        self.return_context_observation = not (
            learn_sub_trajectories or self.do_replanning)
        self.traj_gen_action_space = self._get_traj_gen_action_space()
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()

        # rendering
        self.do_render = False
        self.verbose = verbose

        # condition value
        self.condition_on_desired = condition_on_desired
        self.condition_pos = None
        self.condition_vel = None

        self.max_planning_times = max_planning_times
        self.plan_steps = 0

    def observation(self, observation):
        # return context space if we are
        if self.return_context_observation:
            observation = observation[self.env.context_mask]
        # cast dtype because metaworld returns incorrect that throws gym error
        return observation.astype(self.observation_space.dtype)

    def get_trajectory(self, action: np.ndarray) -> Tuple:
        duration = self.duration
        if self.learn_sub_trajectories:
            duration = None
            # reset  with every new call as we need to set all arguments, such as tau, delay, again.
            # If we do not do this, the traj_gen assumes we are continuing the trajectory.
            self.traj_gen.reset()

        clipped_params = np.clip(
            action, self.traj_gen_action_space.low, self.traj_gen_action_space.high)
        self.traj_gen.set_params(clipped_params)
        init_time = np.array(
            0 if not self.do_replanning else self.current_traj_steps * self.dt)

        condition_pos = self.condition_pos if self.condition_pos is not None else self.env.get_wrapper_attr('current_pos')
        condition_vel = self.condition_vel if self.condition_vel is not None else self.env.get_wrapper_attr('current_vel')

        self.traj_gen.set_initial_conditions(
            init_time, condition_pos, condition_vel)
        self.traj_gen.set_duration(duration, self.dt)

        position = get_numpy(self.traj_gen.get_traj_pos())
        velocity = get_numpy(self.traj_gen.get_traj_vel())

        return position, velocity

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

        position, velocity = self.get_trajectory(action)
        position, velocity = self.env.set_episode_arguments(action, position, velocity)
        traj_is_valid, position, velocity = self.env.preprocessing_and_validity_callback(action, position, velocity,
                                                                                         self.tau_bound, self.delay_bound)

        trajectory_length = len(position)
        rewards = np.zeros(shape=(trajectory_length,))
        if self.verbose >= 2:
            actions = np.zeros(shape=(trajectory_length,) +
                               self.env.action_space.shape)
            observations = np.zeros(shape=(trajectory_length,) + self.env.observation_space.shape,
                                    dtype=self.env.observation_space.dtype)

        infos = dict()
        terminated, truncated = False, False

        if not traj_is_valid:
            obs, trajectory_return, terminated, truncated, infos = self.env.invalid_traj_callback(action, position, velocity,
                                                                                                  self.return_context_observation, self.tau_bound, self.delay_bound)
            return self.observation(obs), trajectory_return, terminated, truncated, infos

        self.plan_steps += 1
        for t, (pos, vel) in enumerate(zip(position, velocity)):
            step_action = self.tracking_controller.get_action(
                pos, vel, self.env.get_wrapper_attr('current_pos'), self.env.get_wrapper_attr('current_vel'))
            c_action = np.clip(
                step_action, self.env.action_space.low, self.env.action_space.high)
            obs, c_reward, terminated, truncated, info = self.env.step(
                c_action)
            rewards[t] = c_reward

            if self.verbose >= 2:
                actions[t, :] = c_action
                observations[t, :] = obs

            for k, v in info.items():
                elems = infos.get(k, [None] * trajectory_length)
                elems[t] = v
                infos[k] = elems

            if self.do_render:
                self.env.render()


            if terminated or truncated or (self.replanning_schedule(self.env.get_wrapper_attr('current_pos'), self.env.get_wrapper_attr('current_vel'), obs, c_action, t + 1 + self.current_traj_steps) and self.plan_steps < self.max_planning_times):

                if self.condition_on_desired:
                    self.condition_pos = pos
                    self.condition_vel = vel

                break

        infos.update({k: v[:t + 1] for k, v in infos.items()})
        self.current_traj_steps += t + 1

        if self.verbose >= 2:
            infos['positions'] = position
            infos['velocities'] = velocity
            infos['step_actions'] = actions[:t + 1]
            infos['step_observations'] = observations[:t + 1]
            infos['step_rewards'] = rewards[:t + 1]

        infos['trajectory_length'] = t + 1
        trajectory_return = self.reward_aggregation(rewards[:t + 1])
        return self.observation(obs), trajectory_return, terminated, truncated, infos

    def render(self):
        self.do_render = True

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) \
            -> Tuple[ObsType, Dict[str, Any]]:
        self.current_traj_steps = 0
        self.plan_steps = 0
        self.traj_gen.reset()
        self.condition_pos = None
        self.condition_vel = None
        return super(BlackBoxWrapper, self).reset(seed=seed, options=options)
