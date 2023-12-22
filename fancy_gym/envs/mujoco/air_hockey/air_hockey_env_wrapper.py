from copy import deepcopy
import numpy as np
from gymnasium import spaces

import fancy_gym.envs.mujoco.air_hockey.constraints as constraints
from fancy_gym.envs.mujoco.air_hockey import position_control_wrapper as position
from fancy_gym.envs.mujoco.air_hockey.utils import robot_to_world
from mushroom_rl.core import Environment

class AirHockeyEnv(Environment):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, env_mode=None, interpolation_order=3, render_mode=None, width=1920, height=1080, **kwargs):
        """
        Environment Constructor

        Args:
            env [string]:
                The string to specify the running environments. Available environments: [3dof-hit, 3dof-defend, 7dof-hit, 7dof-defend, tournament].
            interpolation_order (int, 3): Type of interpolation used, has to correspond to action shape. Order 1-5 are
                    polynomial interpolation of the degree. Order -1 is linear interpolation of position and velocity.
                    Set Order to None in order to turn off interpolation. In this case the action has to be a trajectory
                    of position, velocity and acceleration of the shape (20, 3, n_joints)
        """

        env_dict = {
            "tournament": position.IiwaPositionTournament,

            "7dof-hit": position.IiwaPositionHit,
            "7dof-defend": position.IiwaPositionDefend,

            "3dof-hit": position.PlanarPositionHit,
            "3dof-defend": position.PlanarPositionDefend,

            "7dof-hit-airhockit2023": position.IiwaPositionHitAirhocKIT2023,
            "7dof-defend-airhockit2023": position.IiwaPositionDefendAirhocKIT2023,
        }

        if env_mode not in env_dict:
            raise Exception(f"Please specify one of the environments in {list(env_dict.keys())} for env_mode parameter!")

        if env_mode == "tournament" and type(interpolation_order) != tuple:
            interpolation_order = (interpolation_order, interpolation_order)

        self.render_mode = render_mode
        self.render_human_active = False

        # Determine headless mode based on render_mode
        headless = self.render_mode == 'rgb_array'
        
        # Prepare viewer_params
        viewer_params = kwargs.get('viewer_params', {})
        viewer_params.update({'headless': headless, 'width': width, 'height': height})
        kwargs['viewer_params'] = viewer_params

        self.base_env = env_dict[env_mode](interpolation_order=interpolation_order, **kwargs)
        self.env_name = env_mode
        self.env_info = self.base_env.env_info

        if hasattr(self.base_env, "wrapper_obs_space") and hasattr(self.base_env, "wrapper_act_space"):
            self.observation_space = self.base_env.wrapper_obs_space
            self.action_space = self.base_env.wrapper_act_space
        else:
            single_robot_obs_size = len(self.base_env.info.observation_space.low)
            if env_mode == "tournament":
                self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,single_robot_obs_size), dtype=np.float64)
            else:
                self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(single_robot_obs_size,), dtype=np.float64)
            robot_info = self.env_info["robot"]

            if env_mode != "tournament":
                if interpolation_order in [1, 2]:
                    self.action_space = spaces.Box(low=robot_info["joint_pos_limit"][0], high=robot_info["joint_pos_limit"][1])
                if interpolation_order in [3, 4, -1]:
                    self.action_space = spaces.Box(low=np.vstack([robot_info["joint_pos_limit"][0], robot_info["joint_vel_limit"][0]]), 
                                                high=np.vstack([robot_info["joint_pos_limit"][1], robot_info["joint_vel_limit"][1]]))
                if interpolation_order in [5]:
                    self.action_space = spaces.Box(low=np.vstack([robot_info["joint_pos_limit"][0], robot_info["joint_vel_limit"][0], robot_info["joint_acc_limit"][0]]), 
                                                high=np.vstack([robot_info["joint_pos_limit"][1], robot_info["joint_vel_limit"][1], robot_info["joint_acc_limit"][1]]))
            else:
                acts = [None, None]
                for i in range(2):
                    if interpolation_order[i] in [1, 2]:
                        acts[i] = spaces.Box(low=robot_info["joint_pos_limit"][0], high=robot_info["joint_pos_limit"][1])
                    if interpolation_order[i] in [3, 4, -1]:
                        acts[i] = spaces.Box(low=np.vstack([robot_info["joint_pos_limit"][0], robot_info["joint_vel_limit"][0]]), 
                                                high=np.vstack([robot_info["joint_pos_limit"][1], robot_info["joint_vel_limit"][1]]))
                    if interpolation_order[i] in [5]:
                        acts[i] = spaces.Box(low=np.vstack([robot_info["joint_pos_limit"][0], robot_info["joint_vel_limit"][0], robot_info["joint_acc_limit"][0]]), 
                                                high=np.vstack([robot_info["joint_pos_limit"][1], robot_info["joint_vel_limit"][1], robot_info["joint_acc_limit"][1]]))
                self.action_space = spaces.Tuple((acts[0], acts[1]))

        constraint_list = constraints.ConstraintList()
        constraint_list.add(constraints.JointPositionConstraint(self.env_info))
        constraint_list.add(constraints.JointVelocityConstraint(self.env_info))
        constraint_list.add(constraints.EndEffectorConstraint(self.env_info))
        if "7dof" in self.env_name or self.env_name == "tournament":
            constraint_list.add(constraints.LinkConstraint(self.env_info))

        self.env_info['constraints'] = constraint_list
        self.env_info['env_name'] = self.env_name

        super().__init__(self.base_env.info)

    def step(self, action):
        obs, reward, done, info = self.base_env.step(action)

        if "tournament" in self.env_name:
            info["constraints_value"] = list()
            info["jerk"] = list()
            for i in range(2):
                obs_agent = obs[i * int(len(obs) / 2): (i + 1) * int(len(obs) / 2)]
                info["constraints_value"].append(deepcopy(self.env_info['constraints'].fun(
                    obs_agent[self.env_info['joint_pos_ids']], obs_agent[self.env_info['joint_vel_ids']])))
                info["jerk"].append(
                    self.base_env.jerk[i * self.env_info['robot']['n_joints']:(i + 1) * self.env_info['robot'][
                        'n_joints']])

            info["score"] = self.base_env.score
            info["faults"] = self.base_env.faults

        else:
            info["constraints_value"] = deepcopy(self.env_info['constraints'].fun(obs[self.env_info['joint_pos_ids']],
                                                                                  obs[self.env_info['joint_vel_ids']]))
            info["jerk"] = self.base_env.jerk
            info["success"] = self.check_success(obs)

        if self.env_info['env_name'] == "tournament":
            obs = np.array(np.split(obs, 2))

        if self.render_human_active:
            self.base_env.render()

        return obs, reward, done, False, info

    def render(self):
        if self.render_mode == 'rgb_array':
            return self.base_env.render(record = True)
        elif self.render_mode == 'human':
            self.render_human_active = True
            self.base_env.render()
        else:
            raise ValueError(f"Unsupported render mode: '{self.render_mode}'")
            
    def reset(self, seed=None, options={}):
        self.base_env.seed(seed)
        obs = self.base_env.reset()
        if self.env_info['env_name'] == "tournament":
            obs = np.array(np.split(obs, 2))
        return obs, {}

    def check_success(self, obs):
        puck_pos, puck_vel = self.base_env.get_puck(obs)

        puck_pos, _ = robot_to_world(self.base_env.env_info["robot"]["base_frame"][0], translation=puck_pos)
        success = 0

        if "hit" in self.env_name:
            if puck_pos[0] - self.base_env.env_info['table']['length'] / 2 > 0 and \
                    np.abs(puck_pos[1]) - self.base_env.env_info['table']['goal_width'] / 2 < 0:
                success = 1

        elif "defend" in self.env_name:
            if -0.8 < puck_pos[0] <= -0.2 and puck_vel[0] < 0.1:
                success = 1

        elif "prepare" in self.env_name:
            if -0.8 < puck_pos[0] <= -0.2 and np.abs(puck_pos[1]) < 0.39105 and puck_vel[0] < 0.1:
                success = 1
        return success

    @property
    def unwrapped(self):
        return self
    
    def close(self):
        self.base_env.stop()


if __name__ == "__main__":
    env = AirHockeyEnv(env_mode="7dof-hit")
    env.reset()

    R = 0.
    J = 0.
    gamma = 1.
    steps = 0
    while True:
        action = np.random.uniform(-1, 1, (2, env.env_info['robot']['n_joints'])) * 3
        observation, reward, done, info = env.step(action)
        env.render()
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
