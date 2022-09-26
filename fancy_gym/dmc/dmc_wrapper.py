# Adopted from: https://github.com/denisyarats/dmc2gym/blob/master/dmc2gym/wrappers.py
# License: MIT
# Copyright (c) 2020 Denis Yarats
import collections
from collections.abc import MutableMapping
from typing import Any, Dict, Tuple, Optional, Union, Callable

import gym
import numpy as np
from dm_control import composer
from dm_control.rl import control
from dm_env import specs
from gym import spaces
from gym.core import ObsType


def _spec_to_box(spec):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32, \
            f"Only float64 and float32 types are allowed, instead {s.dtype} was found"
        dim = int(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=s.dtype)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=s.dtype)
            return s.minimum + zeros, s.maximum + zeros

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0)
    high = np.concatenate(maxs, axis=0)
    assert low.shape == high.shape
    return spaces.Box(low, high, dtype=s.dtype)


def _flatten_obs(obs: MutableMapping):
    """
    Flattens an observation of type MutableMapping, e.g. a dict to a 1D array.
    Args:
        obs: observation to flatten

    Returns: 1D array of observation

    """

    if not isinstance(obs, MutableMapping):
        raise ValueError(f'Requires dict-like observations structure. {type(obs)} found.')

    # Keep key order consistent for non OrderedDicts
    keys = obs.keys() if isinstance(obs, collections.OrderedDict) else sorted(obs.keys())

    obs_vals = [np.array([obs[key]]) if np.isscalar(obs[key]) else obs[key].ravel() for key in keys]
    return np.concatenate(obs_vals)


class DMCWrapper(gym.Env):
    def __init__(self,
                 env: Callable[[], Union[composer.Environment, control.Environment]],
                 ):

        # TODO: Currently this is required to be a function because dmc does not allow to copy composers environments
        self._env = env()

        # action and observation space
        self._action_space = _spec_to_box([self._env.action_spec()])
        self._observation_space = _spec_to_box(self._env.observation_spec().values())

        self._window = None
        self.id = 'dmc'

    def __getattr__(self, item):
        """Propagate only non-existent properties to wrapped env."""
        if item.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(item))
        if item in self.__dict__:
            return getattr(self, item)
        return getattr(self._env, item)

    def _get_obs(self, time_step):
        obs = _flatten_obs(time_step.observation).astype(self.observation_space.dtype)
        return obs

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def dt(self):
        return self._env.control_timestep()

    def seed(self, seed=None):
        self._action_space.seed(seed)
        self._observation_space.seed(seed)

    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        assert self._action_space.contains(action)
        extra = {'internal_state': self._env.physics.get_state().copy()}

        time_step = self._env.step(action)
        reward = time_step.reward or 0.
        done = time_step.last()
        obs = self._get_obs(time_step)
        extra['discount'] = time_step.discount

        return obs, reward, done, extra

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False,
              options: Optional[dict] = None, ) -> Union[ObsType, Tuple[ObsType, dict]]:
        time_step = self._env.reset()
        obs = self._get_obs(time_step)
        return obs

    def render(self, mode='rgb_array', height=240, width=320, camera_id=-1, overlays=(), depth=False,
               segmentation=False, scene_option=None, render_flag_overrides=None):

        # assert mode == 'rgb_array', 'only support rgb_array mode, given %s' % mode
        if mode == "rgb_array":
            return self._env.physics.render(height=height, width=width, camera_id=camera_id, overlays=overlays,
                                            depth=depth, segmentation=segmentation, scene_option=scene_option,
                                            render_flag_overrides=render_flag_overrides)

        # Render max available buffer size. Larger is only possible by altering the XML.
        img = self._env.physics.render(height=self._env.physics.model.vis.global_.offheight,
                                       width=self._env.physics.model.vis.global_.offwidth,
                                       camera_id=camera_id, overlays=overlays, depth=depth, segmentation=segmentation,
                                       scene_option=scene_option, render_flag_overrides=render_flag_overrides)

        if depth:
            img = np.dstack([img.astype(np.uint8)] * 3)

        if mode == 'human':
            try:
                import cv2
                if self._window is None:
                    self._window = cv2.namedWindow(self.id, cv2.WINDOW_AUTOSIZE)
                cv2.imshow(self.id, img[..., ::-1])  # Image in BGR
                cv2.waitKey(1)
            except ImportError:
                raise gym.error.DependencyNotInstalled("Rendering requires opencv. Run `pip install opencv-python`")
            # PYGAME seems to destroy some global rendering configs from the physics render
            # except ImportError:
            #     import pygame
            #     img_copy = img.copy().transpose((1, 0, 2))
            #     if self._window is None:
            #         pygame.init()
            #         pygame.display.init()
            #         self._window = pygame.display.set_mode(img_copy.shape[:2])
            #         self.clock = pygame.time.Clock()
            #
            #     surf = pygame.surfarray.make_surface(img_copy)
            #     self._window.blit(surf, (0, 0))
            #     pygame.event.pump()
            #     self.clock.tick(30)
            #     pygame.display.flip()

    def close(self):
        super().close()
        if self._window is not None:
            try:
                import cv2
                cv2.destroyWindow(self.id)
            except ImportError:
                import pygame

                pygame.display.quit()
                pygame.quit()

    @property
    def reward_range(self) -> Tuple[float, float]:
        reward_spec = self._env.reward_spec()
        if isinstance(reward_spec, specs.BoundedArray):
            return reward_spec.minimum, reward_spec.maximum
        return -float('inf'), float('inf')

    @property
    def metadata(self):
        return {'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': round(1.0 / self._env.control_timestep())}
