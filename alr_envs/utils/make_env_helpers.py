from alr_envs.utils.wrapper.dmp_wrapper import DmpWrapper
from alr_envs.utils.wrapper.detpmp_wrapper import DetPMPWrapper
import gym
from gym.vector.utils import write_to_shared_memory
import sys


def make_env(env_id, seed, rank):
    env = gym.make(env_id)
    env.seed(seed + rank)
    return lambda: env


def make_contextual_env(env_id, context, seed, rank):
    env = gym.make(env_id, context=context)
    env.seed(seed + rank)
    return lambda: env


def make_dmp_env(**kwargs):
    name = kwargs.pop("name")
    _env = gym.make(name)
    return DmpWrapper(_env, **kwargs)


def make_detpmp_env(**kwargs):
    name = kwargs.pop("name")
    _env = gym.make(name)
    return DetPMPWrapper(_env, **kwargs)


# def _worker(index, env_fn, pipe, parent_pipe, shared_memory, error_queue):
#     assert shared_memory is None
#     env = env_fn()
#     parent_pipe.close()
#     try:
#         while True:
#             command, data = pipe.recv()
#             if command == 'reset':
#                 observation = env.reset()
#                 pipe.send((observation, True))
#             elif command == 'configure':
#                 env.configure(data)
#                 pipe.send((None, True))
#             elif command == 'step':
#                 observation, reward, done, info = env.step(data)
#                 if done:
#                     observation = env.reset()
#                 pipe.send(((observation, reward, done, info), True))
#             elif command == 'seed':
#                 env.seed(data)
#                 pipe.send((None, True))
#             elif command == 'close':
#                 pipe.send((None, True))
#                 break
#             elif command == '_check_observation_space':
#                 pipe.send((data == env.observation_space, True))
#             else:
#                 raise RuntimeError('Received unknown command `{0}`. Must '
#                     'be one of {`reset`, `step`, `seed`, `close`, '
#                     '`_check_observation_space`}.'.format(command))
#     except (KeyboardInterrupt, Exception):
#         error_queue.put((index,) + sys.exc_info()[:2])
#         pipe.send((None, False))
#     finally:
#         env.close()
#
#
# def _worker_shared_memory(index, env_fn, pipe, parent_pipe, shared_memory, error_queue):
#     assert shared_memory is not None
#     env = env_fn()
#     observation_space = env.observation_space
#     parent_pipe.close()
#     try:
#         while True:
#             command, data = pipe.recv()
#             if command == 'reset':
#                 observation = env.reset()
#                 write_to_shared_memory(index, observation, shared_memory,
#                                        observation_space)
#                 pipe.send((None, True))
#             elif command == 'configure':
#                 env.configure(data)
#                 pipe.send((None, True))
#             elif command == 'step':
#                 observation, reward, done, info = env.step(data)
#                 if done:
#                     observation = env.reset()
#                 write_to_shared_memory(index, observation, shared_memory,
#                                        observation_space)
#                 pipe.send(((None, reward, done, info), True))
#             elif command == 'seed':
#                 env.seed(data)
#                 pipe.send((None, True))
#             elif command == 'close':
#                 pipe.send((None, True))
#                 break
#             elif command == '_check_observation_space':
#                 pipe.send((data == observation_space, True))
#             else:
#                 raise RuntimeError('Received unknown command `{0}`. Must '
#                     'be one of {`reset`, `step`, `seed`, `close`, '
#                     '`_check_observation_space`}.'.format(command))
#     except (KeyboardInterrupt, Exception):
#         error_queue.put((index,) + sys.exc_info()[:2])
#         pipe.send((None, False))
#     finally:
#         env.close()


# def viapoint_dmp(**kwargs):
#     _env = gym.make("alr_envs:ViaPointReacher-v0")
#     # _env = ViaPointReacher(**kwargs)
#     return DmpWrapper(_env, num_dof=5, num_basis=5, duration=2, alpha_phase=2.5, dt=_env.dt,
#                       start_pos=_env.start_pos, learn_goal=False, policy_type="velocity", weights_scale=50)
#
#
# def holereacher_dmp(**kwargs):
#     _env = gym.make("alr_envs:HoleReacher-v0")
#     # _env = HoleReacher(**kwargs)
#     return DmpWrapper(_env, num_dof=5, num_basis=5, duration=2, dt=_env.dt, learn_goal=True, alpha_phase=2,
#                       start_pos=_env.start_pos, policy_type="velocity", weights_scale=50, goal_scale=0.1)
#
#
# def holereacher_fix_goal_dmp(**kwargs):
#     _env = gym.make("alr_envs:HoleReacher-v0")
#     # _env = HoleReacher(**kwargs)
#     return DmpWrapper(_env, num_dof=5, num_basis=5, duration=2, dt=_env.dt, learn_goal=False, alpha_phase=2,
#                       start_pos=_env.start_pos, policy_type="velocity", weights_scale=50, goal_scale=1,
#                       final_pos=np.array([2.02669572, -1.25966385, -1.51618198, -0.80946476, 0.02012344]))
#
#
# def holereacher_detpmp(**kwargs):
#     _env = gym.make("alr_envs:HoleReacher-v0")
#     # _env = HoleReacher(**kwargs)
#     return DetPMPWrapper(_env, num_dof=5, num_basis=5, width=0.005, policy_type="velocity", start_pos=_env.start_pos,
#                          duration=2, post_traj_time=0, dt=_env.dt, weights_scale=0.25, zero_start=True, zero_goal=False)
