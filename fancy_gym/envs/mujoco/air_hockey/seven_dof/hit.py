import numpy as np

from fancy_gym.envs.mujoco.air_hockey.seven_dof.env_single import AirHockeySingle
from fancy_gym.envs.mujoco.air_hockey.seven_dof.airhockit_base_env import AirhocKIT2023BaseEnv


class AirHockeyHit(AirHockeySingle):
    """
        Class for the air hockey hitting task.
    """
    def __init__(self, gamma=0.99, horizon=500, moving_init=True, viewer_params={}):
        """
            Constructor
            Args:
                opponent_agent(Agent, None): Agent which controls the opponent
                moving_init(bool, False): If true, initialize the puck with inital velocity.
        """
        super().__init__(gamma=gamma, horizon=horizon, viewer_params=viewer_params)

        self.moving_init = moving_init
        hit_width = self.env_info['table']['width'] / 2 - self.env_info['puck']['radius'] - \
                    self.env_info['mallet']['radius'] * 2
        self.hit_range = np.array([[-0.7, -0.2], [-hit_width, hit_width]])  # Table Frame
        self.init_velocity_range = (0, 0.5)  # Table Frame
        self.init_ee_range = np.array([[0.60, 1.25], [-0.4, 0.4]])  # Robot Frame


    def setup(self, obs):
        # Initial position of the puck
        puck_pos = np.random.rand(2) * (self.hit_range[:, 1] - self.hit_range[:, 0]) + self.hit_range[:, 0]

        self._write_data("puck_x_pos", puck_pos[0])
        self._write_data("puck_y_pos", puck_pos[1])

        if self.moving_init:
            lin_vel = np.random.uniform(self.init_velocity_range[0], self.init_velocity_range[1])
            angle = np.random.uniform(-np.pi / 2 - 0.1, np.pi / 2 + 0.1)
            puck_vel = np.zeros(3)
            puck_vel[0] = -np.cos(angle) * lin_vel
            puck_vel[1] = np.sin(angle) * lin_vel
            puck_vel[2] = np.random.uniform(-2, 2)

            self._write_data("puck_x_vel", puck_vel[0])
            self._write_data("puck_y_vel", puck_vel[1])
            self._write_data("puck_yaw_vel", puck_vel[2])

        super(AirHockeyHit, self).setup(obs)

    def reward(self, state, action, next_state, absorbing):
        return 0

    def is_absorbing(self, obs):
        puck_pos, puck_vel = self.get_puck(obs)
        # Stop if the puck bounces back on the opponents wall
        if puck_pos[0] > 0 and puck_vel[0] < 0:
            return True
        return super(AirHockeyHit, self).is_absorbing(obs)

class AirHockeyHitAirhocKIT2023(AirhocKIT2023BaseEnv):
    def __init__(self, gamma=0.99, horizon=500, moving_init=True, viewer_params={}, **kwargs):
        super().__init__(gamma=gamma, horizon=horizon, viewer_params=viewer_params, **kwargs)

        self.moving_init = moving_init
        hit_width = self.env_info['table']['width'] / 2 - self.env_info['puck']['radius'] - \
                    self.env_info['mallet']['radius'] * 2
        self.hit_range = np.array([[-0.7, -0.2], [-hit_width, hit_width]])  # Table Frame
        self.init_velocity_range = (0, 0.5)  # Table Frame
        self.init_ee_range = np.array([[0.60, 1.25], [-0.4, 0.4]])  # Robot Frame
        self._setup_metrics()

    def reset(self, *args):
        obs = super().reset()
        self.last_ee_pos = self.last_planned_world_pos.copy()
        self.last_ee_pos[0] -= 1.51
        return obs

    def setup(self, obs):
        self._setup_metrics()
        puck_pos = np.random.rand(2) * (self.hit_range[:, 1] - self.hit_range[:, 0]) + self.hit_range[:, 0]

        self._write_data("puck_x_pos", puck_pos[0])
        self._write_data("puck_y_pos", puck_pos[1])

        if self.moving_init:
            lin_vel = np.random.uniform(self.init_velocity_range[0], self.init_velocity_range[1])
            angle = np.random.uniform(-np.pi / 2 - 0.1, np.pi / 2 + 0.1)
            puck_vel = np.zeros(3)
            puck_vel[0] = -np.cos(angle) * lin_vel
            puck_vel[1] = np.sin(angle) * lin_vel
            puck_vel[2] = np.random.uniform(-2, 2)

            self._write_data("puck_x_vel", puck_vel[0])
            self._write_data("puck_y_vel", puck_vel[1])
            self._write_data("puck_yaw_vel", puck_vel[2])

        super().setup(obs)

    def _setup_metrics(self):
        self.episode_steps = 0
        self.has_scored = False

    def _step_finalize(self):
        cur_obs = self._create_observation(self.obs_helper._build_obs(self._data))
        puck_pos, _ = self.get_puck(cur_obs)  # world frame [x, y, z] and [x', y', z']

        if not self.has_scored:
            boundary = np.array([self.env_info['table']['length'], self.env_info['table']['width']]) / 2
            self.has_scored = np.any(np.abs(puck_pos[:2]) > boundary) and puck_pos[0] > 0

        self.episode_steps += 1
        return super()._step_finalize()
    
    def reward(self, state, action, next_state, absorbing):
        rew = 0
        puck_pos, puck_vel = self.get_puck(next_state)
        ee_pos, _ = self.get_ee()
        ee_vel = (ee_pos - self.last_ee_pos) / 0.02
        self.last_ee_pos = ee_pos

        if puck_vel[0] < 0.25 and puck_pos[0] < 0:
            ee_puck_dir = (puck_pos - ee_pos)[:2]
            ee_puck_dir = ee_puck_dir / np.linalg.norm(ee_puck_dir)
            rew += 1 * max(0, np.dot(ee_puck_dir, ee_vel[:2]))
        else:
            rew += 10 * np.linalg.norm(puck_vel[:2])

        if self.has_scored:
            rew += 2000 + 5000 * np.linalg.norm(puck_vel[:2])
        
        return rew

    def is_absorbing(self, obs):
        puck_pos, puck_vel = self.get_puck(obs)
        # Stop if the puck bounces back on the opponents wall
        if puck_pos[0] > 0 and puck_vel[0] < 0:
            return True

        if self.has_scored:
            return True

        if self.episode_steps == self._mdp_info.horizon:
            return True

        return super().is_absorbing(obs)


if __name__ == '__main__':
    env = AirHockeyHit(moving_init=True)
    env.reset()

    steps = 0
    while True:
        action = np.zeros(7)

        observation, reward, done, info = env.step(action)
        env.render()
        if done or steps > env.info.horizon:
            steps = 0
            env.reset()
