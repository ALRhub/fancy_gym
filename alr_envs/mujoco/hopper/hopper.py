from gym import utils
import numpy as np
from alr_envs.mujoco import alr_mujoco_env

class hopper(alr_mujoco_env.AlrMujocoEnv):
    __init__(self):


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