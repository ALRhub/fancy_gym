import numpy as np

from fancy_gym.envs.mujoco.air_hockey.seven_dof.env_single import AirHockeySingle
from fancy_gym.envs.mujoco.air_hockey.seven_dof.airhockit_base_env import AirhocKIT2023BaseEnv


class AirHockeyDefend(AirHockeySingle):
    """
        Class for the air hockey defending task.
        The agent should stop the puck at the line x=-0.6.
    """
    def __init__(self, gamma=0.99, horizon=500, viewer_params={}):
        self.init_velocity_range = (1, 3)
        self.start_range = np.array([[0.29, 0.65], [-0.4, 0.4]])  # Table Frame
        super().__init__(gamma=gamma, horizon=horizon, viewer_params=viewer_params)

    def setup(self, obs):
        puck_pos = np.random.rand(2) * (self.start_range[:, 1] - self.start_range[:, 0]) + self.start_range[:, 0]

        lin_vel = np.random.uniform(self.init_velocity_range[0], self.init_velocity_range[1])
        angle = np.random.uniform(-0.5, 0.5)

        puck_vel = np.zeros(3)
        puck_vel[0] = -np.cos(angle) * lin_vel
        puck_vel[1] = np.sin(angle) * lin_vel
        puck_vel[2] = np.random.uniform(-10, 10)

        self._write_data("puck_x_pos", puck_pos[0])
        self._write_data("puck_y_pos", puck_pos[1])
        self._write_data("puck_x_vel", puck_vel[0])
        self._write_data("puck_y_vel", puck_vel[1])
        self._write_data("puck_yaw_vel", puck_vel[2])

        super().setup(obs)

    def reward(self, state, action, next_state, absorbing):
        return 0

    def is_absorbing(self, state):
        puck_pos, puck_vel = self.get_puck(state)
        # If puck is over the middle line and moving towards opponent
        if puck_pos[0] > 0 and puck_vel[0] > 0:
            return True
        if np.linalg.norm(puck_vel[:2]) < 0.1:
            return True
        return super().is_absorbing(state)

class AirHockeyDefendAirhocKIT2023(AirhocKIT2023BaseEnv):
    def __init__(self, gamma=0.99, horizon=200, viewer_params={}, **kwargs):
        super().__init__(gamma=gamma, horizon=horizon, viewer_params=viewer_params, **kwargs)
        self.init_velocity_range = (1, 3)
        self.start_range = np.array([[0.4, 0.75], [-0.4, 0.4]])  # Table Frame
        self._setup_metrics()
        
    def setup(self, obs):
        self._setup_metrics()
        puck_pos = np.random.rand(2) * (self.start_range[:, 1] - self.start_range[:, 0]) + self.start_range[:, 0]

        lin_vel = np.random.uniform(self.init_velocity_range[0], self.init_velocity_range[1])
        angle = np.random.uniform(-0.5, 0.5)

        puck_vel = np.zeros(3)
        puck_vel[0] = -np.cos(angle) * lin_vel
        puck_vel[1] = np.sin(angle) * lin_vel
        puck_vel[2] = np.random.uniform(-10, 10)

        self._write_data("puck_x_pos", puck_pos[0])
        self._write_data("puck_y_pos", puck_pos[1])
        self._write_data("puck_x_vel", puck_vel[0])
        self._write_data("puck_y_vel", puck_vel[1])
        self._write_data("puck_yaw_vel", puck_vel[2])

        super().setup(obs)

    def reset(self, *args):
        obs = super().reset()
        self.hit_step_flag = False
        self.hit_step = False
        self.received_hit_reward = False
        self.give_reward_next = False
        return obs

    def _setup_metrics(self):
        self.episode_steps = 0
        self.has_hit = False

    def _simulation_post_step(self):
        if not self.has_hit:
            self.has_hit = self._check_collision("puck", "robot_1/ee")

        super()._simulation_post_step()

    def _step_finalize(self):
        self.episode_steps += 1
        return super()._step_finalize()
    
    def reward(self, state, action, next_state, absorbing):
        puck_pos, puck_vel = self.get_puck(next_state)
        ee_pos, _ = self.get_ee()
        rew = 0.01
        if -0.7 < puck_pos[0] <= -0.2 and np.linalg.norm(puck_vel[:2]) < 0.1:
            assert absorbing
            rew += 70

        if self.has_hit and not self.hit_step_flag:
            self.hit_step_flag = True
            self.hit_step = True
        else:
            self.hit_step = False

        f = lambda puck_vel: 30 + 100 * (100 ** (-0.25 * np.linalg.norm(puck_vel[:2])))
        if not self.give_reward_next and not self.received_hit_reward and self.hit_step and ee_pos[0] < puck_pos[0]:
            self.hit_this_step = True
            if np.linalg.norm(puck_vel[:2]) < 0.1:
                return rew + f(puck_vel)
            self.give_reward_next = True
            return rew

        if not self.received_hit_reward and self.give_reward_next:
            self.received_hit_reward = True
            if puck_vel[0] >= -0.2:
                return rew + f(puck_vel)
            return rew
        else:
            return rew

    def is_absorbing(self, obs):
        puck_pos, puck_vel = self.get_puck(obs)
        # If puck is over the middle line and moving towards opponent
        if puck_pos[0] > 0 and puck_vel[0] > 0:
            return True

        if self.episode_steps == self._mdp_info.horizon:
            return True
            
        if np.linalg.norm(puck_vel[:2]) < 0.1:
            return True
        return super().is_absorbing(obs)


if __name__ == '__main__':
    env = AirHockeyDefend()

    R = 0.
    J = 0.
    gamma = 1.
    steps = 0
    env.reset()
    env.render()
    while True:
        # action = np.random.uniform(-1, 1, env.info.action_space.low.shape) * 8
        action = np.zeros(7)
        observation, reward, done, info = env.step(action)
        env.render()
        print(observation)
        gamma *= env.info.gamma
        J += gamma * reward
        R += reward
        steps += 1
        if done or steps > env.info.horizon:
            print("J: ", J, " R: ", R)
            R = 0.
            J = 0.
            gamma = 1.
            steps = 0
            env.reset()
