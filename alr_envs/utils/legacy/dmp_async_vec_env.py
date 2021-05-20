import gym
from gym.error import (AlreadyPendingCallError, NoAsyncCallError)
from gym.vector.utils import concatenate, create_empty_array
from gym.vector.async_vector_env import AsyncState
import numpy as np
import multiprocessing as mp
import sys


def _worker(index, env_fn, pipe, parent_pipe, shared_memory, error_queue):
    assert shared_memory is None
    env = env_fn()
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == 'reset':
                observation = env.reset()
                pipe.send((observation, True))
            elif command == 'step':
                observation, reward, done, info = env.step(data)
                if done:
                    observation = env.reset()
                pipe.send(((observation, reward, done, info), True))
            elif command == 'rollout':
                rewards = []
                infos = []
                for p, c in zip(*data):
                    reward, info = env.rollout(p, c)
                    rewards.append(reward)
                    infos.append(info)
                pipe.send(((rewards, infos), (True, ) * len(rewards)))
            elif command == 'seed':
                env.seed(data)
                pipe.send((None, True))
            elif command == 'close':
                env.close()
                pipe.send((None, True))
                break
            elif command == 'idle':
                pipe.send((None, True))
            elif command == '_check_observation_space':
                pipe.send((data == env.observation_space, True))
            else:
                raise RuntimeError('Received unknown command `{0}`. Must '
                    'be one of {`reset`, `step`, `seed`, `close`, '
                    '`_check_observation_space`}.'.format(command))
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()


class DmpAsyncVectorEnv(gym.vector.AsyncVectorEnv):
    def __init__(self, env_fns, n_samples, observation_space=None, action_space=None,
                 shared_memory=False, copy=True, context="spawn", daemon=True, worker=_worker):
        super(DmpAsyncVectorEnv, self).__init__(env_fns,
                                                observation_space=observation_space,
                                                action_space=action_space,
                                                shared_memory=shared_memory,
                                                copy=copy,
                                                context=context,
                                                daemon=daemon,
                                                worker=worker)

        # we need to overwrite the number of samples as we may sample more than num_envs
        self.observations = create_empty_array(self.single_observation_space,
                                               n=n_samples,
                                               fn=np.zeros)

    def __call__(self, params, contexts=None):
        return self.rollout(params, contexts)

    def rollout_async(self, params, contexts):
        """
        Parameters
        ----------
        params : iterable of samples from `action_space`
            List of actions.
        """
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError('Calling `rollout_async` while waiting '
                                          'for a pending call to `{0}` to complete.'.format(
                self._state.value), self._state.value)

        params = np.atleast_2d(params)
        split_params = np.array_split(params, np.minimum(len(params), self.num_envs))
        if contexts is None:
            split_contexts = np.array_split([None, ] * len(params), np.minimum(len(params), self.num_envs))
        else:
            split_contexts = np.array_split(contexts, np.minimum(len(contexts), self.num_envs))

        assert np.all([len(p) == len(c) for p, c in zip(split_params, split_contexts)])
        for pipe, param, context in zip(self.parent_pipes, split_params, split_contexts):
            pipe.send(('rollout', (param, context)))
        for pipe in self.parent_pipes[len(split_params):]:
            pipe.send(('idle', None))
        self._state = AsyncState.WAITING_ROLLOUT

    def rollout_wait(self, timeout=None):
        """
        Parameters
        ----------
        timeout : int or float, optional
            Number of seconds before the call to `step_wait` times out. If
            `None`, the call to `step_wait` never times out.

        Returns
        -------
        observations : sample from `observation_space`
            A batch of observations from the vectorized environment.

        rewards : `np.ndarray` instance (dtype `np.float_`)
            A vector of rewards from the vectorized environment.

        dones : `np.ndarray` instance (dtype `np.bool_`)
            A vector whose entries indicate whether the episode has ended.

        infos : list of dict
            A list of auxiliary diagnostic information.
        """
        self._assert_is_running()
        if self._state != AsyncState.WAITING_ROLLOUT:
            raise NoAsyncCallError('Calling `rollout_wait` without any prior call '
                                   'to `rollout_async`.', AsyncState.WAITING_ROLLOUT.value)

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError('The call to `rollout_wait` has timed out after '
                                  '{0} second{1}.'.format(timeout, 's' if timeout > 1 else ''))

        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        results = [r for r in results if r is not None]
        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT

        rewards, infos = [_flatten_list(r) for r in zip(*results)]

        # for now, we ignore the observations and only return the rewards

        # if not self.shared_memory:
        #     self.observations = concatenate(observations_list, self.observations,
        #                                     self.single_observation_space)

        # return (deepcopy(self.observations) if self.copy else self.observations,
        #         np.array(rewards), np.array(dones, dtype=np.bool_), infos)

        return np.array(rewards), infos

    def rollout(self, actions, contexts):
        self.rollout_async(actions, contexts)
        return self.rollout_wait()


def _flatten_obs(obs):
    assert isinstance(obs, (list, tuple))
    assert len(obs) > 0

    if isinstance(obs[0], dict):
        keys = obs[0].keys()
        return {k: np.stack([o[k] for o in obs]) for k in keys}
    else:
        return np.stack(obs)


def _flatten_list(l):
    assert isinstance(l, (list, tuple))
    assert len(l) > 0
    assert all([len(l_) > 0 for l_ in l])

    return [l__ for l_ in l for l__ in l_]