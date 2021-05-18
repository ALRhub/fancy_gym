from gym import utils
import gym
import numpy as np
from alr_envs.mujoco import alr_mujoco_env
from gym.envs.mujoco import hopper

class hopper(alr_mujoco_env.AlrMujocoEnv):
    def __init__(self):
        alr_mujoco_env.AlrMujocoEnv.__init__(self)


class alr_hopper(hopper):
    def step(self, a):
        heightbefore = self.sim.data.qpos[1]
        self.do_simulation(a, self.frame_skip)
        pos, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (height - heightbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()
        return ob, reward, done, {}

        # posbefore = self.sim.data.qpos[0]
        # self.do_simulation(a, self.frame_skip)
        # posafter, height, ang = self.sim.data.qpos[0:3]
        # alive_bonus = 1.0
        # reward = (posafter - posbefore) / self.dt
        # reward += alive_bonus
        # reward -= 1e-3 * np.square(a).sum()
        # s = self.state_vector()
        # done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
        #             (height > .7) and (abs(ang) < .2))
        # ob = self._get_obs()
        # return ob, reward, done, {}


def example_hopper():
    env = gym.make('Hopper-v2')
    env = hopper
    rewards = 0
    obs = env.reset()

    # number of environment steps
    for i in range(10000):
        obs, reward, done, info = env.step(env.action_space.sample())
        rewards += reward

        if i % 1 == 0:
            env.render()

        if done:
            print(rewards)
            rewards = 0
            obs = env.reset()


if __name__ == "__main__":
    # env = ALRBallInACupEnv()
    # ctxt = np.array([-0.20869846, -0.66376693, 1.18088501])

    # env.configure(ctxt)
    # env.reset()
    # # env.render()
    # for i in range(16000):
    #     # test with random actions
    #     ac = 0.001 * env.action_space.sample()[0:7]
    #     # ac = env.start_pos
    #     # ac[0] += np.pi/2
    #     obs, rew, d, info = env.step(ac)
    #     # env.render()

    #     print(rew)

    #     if d:
    #         break

    # env.close()
    # example_hopper()
    env = gym.make("Hopper-v2")
    print(env.action_space)
    print(env.action_space.shape)
    print(env.observation_space)
    print(env.observation_space.shape)
