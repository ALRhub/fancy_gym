# Adopted from: https://github.com/denisyarats/dmc2gym/blob/master/dmc2gym/wrappers.py
# License: MIT
# Copyright (c) 2020 Denis Yarats
from typing import Any, Dict, Tuple

import numpy as np
from dm_control import manipulation, suite
from dm_env import specs
from gym import core, spaces


def _spec_to_box(spec):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32, f"Only float64 and float32 types are allowed, instead {s.dtype} was found"
        dim = int(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0)
    high = np.concatenate(maxs, axis=0)
    assert low.shape == high.shape
    return spaces.Box(low, high, dtype=np.float32)


def _flatten_obs(obs):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)


class DMCWrapper(core.Env):
    def __init__(
            self,
            domain_name,
            task_name,
            task_kwargs={},
            visualize_reward=True,
            from_pixels=False,
            height=84,
            width=84,
            camera_id=0,
            frame_skip=1,
            environment_kwargs=None,
            channels_first=True
    ):
        assert 'random' in task_kwargs, 'please specify a seed, for deterministic behaviour'
        self._from_pixels = from_pixels
        self._height = height
        self._width = width
        self._camera_id = camera_id
        self._frame_skip = frame_skip
        self._channels_first = channels_first

        # create task
        if domain_name == "manipulation":
            assert not from_pixels, \
                "TODO: Vision interface for manipulation is different to suite and needs to be implemented"
            self._env = manipulation.load(environment_name=task_name, seed=task_kwargs['random'])
        else:
            self._env = suite.load(domain_name=domain_name, task_name=task_name, task_kwargs=task_kwargs,
                                   visualize_reward=visualize_reward, environment_kwargs=environment_kwargs)

        # action and observation space
        self._action_space = _spec_to_box([self._env.action_spec()])
        self._observation_space = _spec_to_box(self._env.observation_spec().values())

        self._last_observation = None
        self.viewer = None

        # set seed
        self.seed(seed=task_kwargs.get('random', 1))

    def __getattr__(self, name):
        """Delegate attribute access to underlying environment."""
        return getattr(self._env, name)

    def _get_obs(self, time_step):
        if self._from_pixels:
            obs = self.render(
                mode="rgb_array",
                height=self._height,
                width=self._width,
                camera_id=self._camera_id
            )
            if self._channels_first:
                obs = obs.transpose(2, 0, 1).copy()
        else:
            obs = _flatten_obs(time_step.observation)
        return obs

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def seed(self, seed=None):
        self._action_space.seed(seed)
        self._observation_space.seed(seed)

    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        assert self._action_space.contains(action)
        reward = 0
        extra = {'internal_state': self._env.physics.get_state().copy()}

        for _ in range(self._frame_skip):
            time_step = self._env.step(action)
            reward += time_step.reward or 0.
            done = time_step.last()
            if done:
                break

        self._last_observation = _flatten_obs(time_step.observation)
        obs = self._get_obs(time_step)
        extra['discount'] = time_step.discount
        return obs, reward, done, extra

    def reset(self) -> np.ndarray:
        time_step = self._env.reset()
        self._last_observation = _flatten_obs(time_step.observation)
        obs = self._get_obs(time_step)
        return obs

    def render(self, mode='rgb_array', height=None, width=None, camera_id=0):
        if self._last_observation is None:
            raise ValueError('Environment not ready to render. Call reset() first.')

        # assert mode == 'rgb_array', 'only support rgb_array mode, given %s' % mode
        if mode == "rgb_array":
            height = height or self._height
            width = width or self._width
            camera_id = camera_id or self._camera_id
            return self._env.physics.render(height=height, width=width, camera_id=camera_id)

        elif mode == 'human':
            if self.viewer is None:
                # pylint: disable=import-outside-toplevel
                # pylint: disable=g-import-not-at-top
                from gym.envs.classic_control import rendering
                self.viewer = rendering.SimpleImageViewer()
            # Render max available buffer size. Larger is only possible by altering the XML.
            img = self._env.physics.render(height=self._env.physics.model.vis.global_.offheight,
                                           width=self._env.physics.model.vis.global_.offwidth)
            self.viewer.imshow(img)
            return self.viewer.isopen

    def close(self):
        super().close()
        if self.viewer is not None and self.viewer.isopen:
            self.viewer.close()

    @property
    def reward_range(self) -> Tuple[float, float]:
        reward_spec = self._env.reward_spec()
        if isinstance(reward_spec, specs.BoundedArray):
            return reward_spec.minimum, reward_spec.maximum
        return -float('inf'), float('inf')
