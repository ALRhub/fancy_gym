from gym import utils
import gym
import numpy as np
from alr_envs.mujoco import alr_mujoco_env
from gym.envs.mujoco import HopperEnv

# class ALRHopperEpisodicEnv(HopperEnv):
#     metadata = {'render.modes': ['human']}


#     def __init__(self):
#         self.heights = [0]
#         self.curr_step = 0
#         self.max_episode_steps = 200
#         super().__init__()
        

#     def step(self, a):
#         heightbefore = self.sim.data.qpos[1]
#         self.do_simulation(a, self.frame_skip)
#         pos, height, angle = self.sim.data.qpos[0:3]

#         self.heights.append(height)
#         # self._max_episode_steps von wrapper
#         reward = 0
#         if (self.curr_step >= self.max_episode_steps-1): # at end of episode get reward for heighest z-value
#             reward = np.max(self.heights)
            
#         s = self.state_vector()
#         done = (not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() 
#                     and (height > .7)))
#         # done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
#         #              (height > .7) and (abs(angle) < .2))
#         obs = self._get_obs()
#         return obs, reward, done, {}

#     def reset_model(self):
#         qpos = self.init_qpos #+ self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
#         qvel = self.init_qvel #+ self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
#         self.set_state(qpos, qvel)
#         return self._get_obs()


        
class ALRHopperEpisodicEnv(alr_mujoco_env.AlrMujocoEnv, utils.EzPickle):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.heights = [0]
        self.curr_step = 0
        self.max_episode_steps = 200
        xml = 'C:/ProgramData/Anaconda3/Lib/site-packages/gym/envs/mujoco/assets/hopper.xml'
        alr_mujoco_env.AlrMujocoEnv.__init__(self, xml, 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        heightbefore = self.sim.data.qpos[1]
        self.do_simulation(a, self.frame_skip)
        pos, height, angle = self.sim.data.qpos[0:3]

        self.heights.append(height)
        # self._max_episode_steps von wrapper
        reward = 0
        if (self.curr_step >= self.max_episode_steps-1): # at end of episode get reward for heighest z-value
            reward = np.max(self.heights)
            
        s = self.state_vector()
        done = (not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() 
                    and (height > .7)))
        # done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
        #              (height > .7) and (abs(angle) < .2))
        obs = self._get_obs()
        return obs, reward, done, {}

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
        


