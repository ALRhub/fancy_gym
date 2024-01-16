from typing import Tuple, Optional, Callable, Dict, Any, Union

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.core import ObsType
from mp_pytorch.mp.mp_interfaces import MPInterface

from fancy_gym.black_box.controller.base_controller import BaseController
from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper
from fancy_gym.utils.utils import get_numpy


class BlackBoxWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env: RawInterfaceWrapper,
        trajectory_generator: MPInterface,
        tracking_controller: BaseController,
        duration: float,
        verbose: int = 1,
        learn_sub_trajectories: bool = False,
        replanning_schedule: Optional[
            Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int], bool]
        ] = None,
        reward_aggregation: Callable[[np.ndarray], float] = np.sum,
        max_planning_times: int = np.inf,
        condition_on_desired: bool = False,
        backend: str = "numpy",
        device: str = "cpu",
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

        self.backend = backend
        self.device = device

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
        self.tau_bound = (
            [-np.inf, np.inf] if self.backend == "numpy" else [-torch.inf, torch.inf]
        )
        self.delay_bound = (
            [-np.inf, np.inf] if self.backend == "numpy" else [-torch.inf, torch.inf]
        )
        if hasattr(self.traj_gen.phase_gn, "tau_bound"):
            self.tau_bound = self.traj_gen.phase_gn.tau_bound
        if hasattr(self.traj_gen.phase_gn, "delay_bound"):
            self.delay_bound = self.traj_gen.phase_gn.delay_bound

        # reward computation
        self.reward_aggregation = reward_aggregation

        # spaces
        self.return_context_observation = not (
            learn_sub_trajectories or self.do_replanning
        )
        self.traj_gen_action_space = self._get_traj_gen_action_space()
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()

        # rendering
        self.render_kwargs = {}
        self.verbose = verbose

        # condition value
        self.condition_on_desired = condition_on_desired
        self.condition_pos = None
        self.condition_vel = None

        self.max_planning_times = max_planning_times
        self.plan_steps = 0

    def observation(self, observation):
        if type(observation) == dict:
            observation = observation["policy"]

        # return context space if we are
        if self.return_context_observation:
            observation = observation[..., self.env.context_mask]

        if self.backend == "numpy":
            # cast dtype because metaworld returns incorrect that throws gym error
            return observation.astype(self.observation_space.dtype)
        else:
            return observation

    def get_trajectory(self, action: Union[np.ndarray, torch.Tensor]) -> Tuple:
        assert (
            len(action.shape) == 2
        ), "action should be a 2D tensor or array, where the first dimension is the batch size"
        batch_size = action.shape[0]

        duration = self.duration
        if self.learn_sub_trajectories:
            duration = None
            # reset  with every new call as we need to set all arguments, such as tau, delay, again.
            # If we do not do this, the traj_gen assumes we are continuing the trajectory.
            self.traj_gen.reset()

        self.traj_gen.set_add_dim(
            [
                batch_size,
            ]
        )

        if self.backend == "torch":
            clipped_params = torch.clamp(
                action,
                torch.tensor(self.traj_gen_action_space.low, device=self.device),
                torch.tensor(self.traj_gen_action_space.high, device=self.device),
            )
        else:
            clipped_params = np.clip(
                action, self.traj_gen_action_space.low, self.traj_gen_action_space.high
            )
        self.traj_gen.set_params(clipped_params)

        init_time = np.array(
            0 if not self.do_replanning else self.current_traj_steps * self.dt
        )
        init_time = np.repeat(init_time, batch_size)

        if self.backend == "torch":
            init_time = torch.tensor(init_time, dtype=torch.float32, device=self.device)

        condition_pos = (
            self.condition_pos
            if self.condition_pos is not None
            else self.env.get_wrapper_attr("current_pos")
        )
        condition_vel = (
            self.condition_vel
            if self.condition_vel is not None
            else self.env.get_wrapper_attr("current_vel")
        )

        if self.backend == "torch":
            condition_pos = (
                condition_pos.unsqueeze(0) if condition_pos.ndim == 1 else condition_pos
            )
            condition_vel = (
                condition_vel.unsqueeze(0) if condition_vel.ndim == 1 else condition_vel
            )
        else:
            condition_pos = (
                np.expand_dims(condition_pos, axis=0)
                if condition_pos.ndim == 1
                else condition_pos
            )
            condition_vel = (
                np.expand_dims(condition_vel, axis=0)
                if condition_vel.ndim == 1
                else condition_vel
            )

        self.traj_gen.set_initial_conditions(init_time, condition_pos, condition_vel)
        self.traj_gen.set_duration(duration, self.dt)

        position = self.traj_gen.get_traj_pos()
        velocity = self.traj_gen.get_traj_vel()

        if self.backend == "numpy":
            position = get_numpy(position)
            velocity = get_numpy(velocity)

        return position, velocity

    def _get_traj_gen_action_space(self):
        """This function can be used to set up an individual space for the parameters of the traj_gen."""
        min_action_bounds, max_action_bounds = self.traj_gen.get_params_bounds()
        action_space = gym.spaces.Box(
            low=min_action_bounds.cpu().numpy(),
            high=max_action_bounds.cpu().numpy(),
            dtype=self.env.action_space.dtype,
        )
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
            return spaces.Box(
                low=min_obs_bound,
                high=max_obs_bound,
                dtype=self.env.observation_space.dtype,
            )
        return self.env.observation_space

    def step(self, action: Union[np.ndarray, torch.Tensor]):
        """This function generates a trajectory based on a MP and then does the usual loop over reset and step"""
        if len(action.shape) == 1:
            action = action.reshape(1, -1)
        batch_size = action.shape[0]

        position, velocity = self.get_trajectory(action)
        position, velocity = self.env.set_episode_arguments(action, position, velocity)
        (
            traj_is_valid,
            position,
            velocity,
        ) = self.env.preprocessing_and_validity_callback(
            action, position, velocity, self.tau_bound, self.delay_bound
        )

        trajectory_length = position.shape[1]
        if self.backend == "torch":
            rewards = torch.zeros(
                size=(
                    batch_size,
                    trajectory_length,
                ),
                device=self.device,
            )
        else:
            rewards = np.zeros(
                shape=(
                    batch_size,
                    trajectory_length,
                )
            )
        if self.verbose >= 2:
            if self.backend == "torch":
                actions = torch.zeros(
                    size=(
                        batch_size,
                        trajectory_length,
                    )
                    + self.env.action_space.shape,
                    device=self.device,
                )
                observations = torch.zeros(
                    size=(
                        batch_size,
                        trajectory_length,
                    )
                    + self.env.observation_space.shape,
                    dtype=self.env.observation_space.dtype,
                    device=self.device,
                )
            else:
                actions = np.zeros(
                    shape=(
                        batch_size,
                        trajectory_length,
                    )
                    + self.env.action_space.shape
                )
                observations = np.zeros(
                    shape=(
                        batch_size,
                        trajectory_length,
                    )
                    + self.env.observation_space.shape,
                    dtype=self.env.observation_space.dtype,
                )

        infos = dict()
        done = False

        if not traj_is_valid:
            (
                obs,
                trajectory_return,
                terminated,
                truncated,
                infos,
            ) = self.env.invalid_traj_callback(
                action,
                position,
                velocity,
                self.return_context_observation,
                self.tau_bound,
                self.delay_bound,
            )
            return (
                self.observation(obs),
                trajectory_return,
                terminated,
                truncated,
                infos,
            )

        self.plan_steps += 1
        for t in range(trajectory_length):
            pos = position[:, t]
            vel = velocity[:, t]

            condition_pos = self.env.get_wrapper_attr("current_pos")
            condition_vel = self.env.get_wrapper_attr("current_vel")

            if self.backend == "torch":
                condition_pos = (
                    condition_pos.unsqueeze(0)
                    if condition_pos.ndim == 1
                    else condition_pos
                )
                condition_vel = (
                    condition_vel.unsqueeze(0)
                    if condition_vel.ndim == 1
                    else condition_vel
                )
            else:
                condition_pos = (
                    np.expand_dims(condition_pos, axis=0)
                    if condition_pos.ndim == 1
                    else condition_pos
                )
                condition_vel = (
                    np.expand_dims(condition_vel, axis=0)
                    if condition_vel.ndim == 1
                    else condition_vel
                )

            step_action = self.tracking_controller.get_action(
                pos, vel, condition_pos, condition_vel
            )
            if self.backend == "torch":
                c_action = torch.clamp(
                    step_action,
                    torch.tensor(self.env.action_space.low, device=self.device),
                    torch.tensor(self.env.action_space.high, device=self.device),
                )
            else:
                c_action = np.clip(
                    step_action, self.env.action_space.low, self.env.action_space.high
                )

            if t == trajectory_length - 1:
                # Orbit resets immidiately after the last step,
                # so only termination and truncation signal are consumed
                dummy_actions = torch.zeros(
                    self.env.unwrapped.action_space.shape, device=self.device
                )
                _, _, terminated, truncated, _ = self.env.step(dummy_actions)
            else:
                # in case env is not vectorized, we need to squeeze the action
                c_action = c_action.squeeze(0) if c_action.shape[0] == 1 else c_action
                obs, c_reward, terminated, truncated, info = self.env.step(c_action)

            rewards[:, t] = c_reward

            if type(obs) == dict:
                obs = obs["policy"]

            if self.verbose >= 2:
                actions[:, t, :] = c_action
                observations[:, t, :] = obs

            for k, v in info.items():
                elems = infos.get(k, [None] * trajectory_length)
                elems[t] = v
                infos[k] = elems

            if self.render_kwargs:
                self.env.render(**self.render_kwargs)

            # TODO: currently only support sync termination and truncation
            if type(truncated) == bool:
                if self.backend == "torch":
                    truncated = torch.tensor(truncated, device=self.device).repeat(
                        batch_size
                    )
                else:
                    truncated = np.array(truncated).repeat(batch_size)

            if type(terminated) == np.ndarray or type(terminated) == torch.Tensor:
                all_terminated = terminated.all()
            if type(truncated) == np.ndarray or type(truncated) == torch.Tensor:
                all_truncated = truncated.all()

            if (
                all_terminated
                or all_truncated
                or (
                    self.replanning_schedule(
                        self.env.get_wrapper_attr("current_pos"),
                        self.env.get_wrapper_attr("current_vel"),
                        obs,
                        c_action,
                        t + 1 + self.current_traj_steps,
                    )
                    and self.plan_steps < self.max_planning_times
                )
            ):
                if self.condition_on_desired:
                    self.condition_pos = pos
                    self.condition_vel = vel

                break

        infos.update({k: v[: t + 1] for k, v in infos.items()})
        self.current_traj_steps += t + 1

        if self.verbose >= 2:
            infos["positions"] = position
            infos["velocities"] = velocity
            infos["step_actions"] = actions[:, : t + 1]
            infos["step_observations"] = observations[:, : t + 1]
            infos["step_rewards"] = rewards[:, : t + 1]

        infos["trajectory_length"] = t + 1
        trajectory_return = self.reward_aggregation(rewards[:, : t + 1], 1)
        return self.observation(obs), trajectory_return, terminated, truncated, infos

    def render(self, **kwargs):
        """Only set render options here, such that they can be used during the rollout.
        This only needs to be called once"""
        self.render_kwargs = kwargs

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[ObsType, Dict[str, Any]]:
        self.current_traj_steps = 0
        self.plan_steps = 0
        self.traj_gen.reset()
        self.condition_pos = None
        self.condition_vel = None
        return super(BlackBoxWrapper, self).reset(seed=seed, options=options)
