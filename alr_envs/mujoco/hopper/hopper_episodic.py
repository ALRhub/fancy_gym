from gym import utils
import os
import numpy as np
from alr_envs.mujoco import alr_mujoco_env

        
class ALRHopperEpisodicEnv(alr_mujoco_env.AlrMujocoEnv, utils.EzPickle):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # xml_path = '/home/schorn/miniconda3/lib/python3.8/site-packages/gym/envs/mujoco/assets/hopper.xml'
        xml_path = 'C:/ProgramData/Anaconda3/Lib/site-packages/gym/envs/mujoco/assets/hopper.xml'
        alr_mujoco_env.AlrMujocoEnv.__init__(self, xml_path, 4)
        utils.EzPickle.__init__(self)      
        self.heights = [0]
        self.curr_step = 0
        # self.max_episode_steps = 200

        self.current_step = 0
        self.max_height = 0

        self._start_pos = np.array([0.0, 0.0, 0.0])
        self._start_vel = np.zeros(3)

        self.max_ctrl = np.array([150., 125., 40.])
        self.p_gains = 1 / self.max_ctrl * np.array([200, 300, 100])
        self.d_gains = 1 / self.max_ctrl * np.array([7, 15, 5])

        self.j_min = np.array([-2.6, -1.985, -2.8])
        self.j_max = np.array([2.6, 1.985, 2.8])

        self.context = None

    # def step(self, a):
    #     heightbefore = self.sim.data.qpos[1]
    #     self.do_simulation(a)
    #     pos, height, angle = self.sim.data.qpos[0:3]

    #     self.heights.append(height)
    #     # self._max_episode_steps von wrapper
    #     reward = 0
    #     if (self.curr_step >= self.max_episode_steps-1): # at end of episode get reward for heighest z-value
    #         reward = np.max(self.heights)
            
    #     s = self.state_vector()
    #     done = (not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() 
    #                 and (height > .7)))
    #     obs = self._get_obs()
    #     return obs, reward, done, {}

    def step(self, a):
        self.current_step += 1
        heightbefore = self.sim.data.qpos[1]
        foot_height_before = self.get_body_com("foot")[2]
        self.do_simulation(a)
        pos, height, angle = self.sim.data.qpos[0:3]
        foot_height = self.get_body_com("foot")[2]
        # check max height of trajectory
        if(height > self.max_height):
            self.max_height = height
        reward = 0
        s = self.state_vector()
        # calculate when its done
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all()
                    and (height > .7))

        # give reward at the end of trajectory
        if(done):
            # calculate reward
            reward = self.max_height
            alive_bonus = 0.1 * self.current_step
            reward += alive_bonus

            # reset variables
            self.max_height = 0
            self.current_step = 0

        obs = self._get_obs()
        return obs, reward, done, {}

    @property
    def start_pos(self):
        return self._start_pos

    @property
    def start_vel(self):
        return self._start_vel

    @property
    def current_pos(self):
        return self.sim.data.qpos[0:3].copy()

    @property
    def current_vel(self):
        return self.sim.data.qvel[0:3].copy()

    def configure(self, context):
        self.context = context
        self.reward_function.reset(context)

    # These functions are for the task with 3 joint actuations
    def extend_des_pos(self, des_pos):
        return self.current_pos

    def extend_des_vel(self, des_vel):
        return self.current_vel

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            np.clip(self.sim.data.qvel.flat, -10, 10)
        ])

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20
        


