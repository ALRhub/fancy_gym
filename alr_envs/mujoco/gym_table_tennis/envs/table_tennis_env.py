import numpy as np
from gym import spaces
from gym.envs.robotics import robot_env, utils
# import xml.etree.ElementTree as ET
from alr_envs.mujoco.gym_table_tennis.utils.rewards.hierarchical_reward import HierarchicalRewardTableTennis
# import glfw
from alr_envs.mujoco.gym_table_tennis.utils.experiment import ball_initialize
from pathlib import Path
import os


class TableTennisEnv(robot_env.RobotEnv):
    """Class for Table Tennis environment.
    """
    def __init__(self, n_substeps=1,
                 model_path=None,
                 initial_qpos=None,
                 initial_ball_state=None,
                 config=None,
                 reward_obj=None
                 ):
        """Initializes a new mujoco based Table Tennis environment.

        Args:
            model_path (string): path to the environments XML file
            initial_qpos (dict): a dictionary of joint names and values that define the initial
            n_actions: Number of joints
            n_substeps (int): number of substeps the simulation runs on every call to step
            scale (double): limit maximum change in position
            initial_ball_state: to reset the ball state
        """
        # self.config = config.config
        if model_path is None:
            path_cws = Path.cwd()
            print(path_cws)
            current_dir = Path(os.path.split(os.path.realpath(__file__))[0])
            table_tennis_env_xml_path = current_dir / "robotics"/"assets"/"table_tennis"/"table_tennis_env.xml"
            model_path = str(table_tennis_env_xml_path)
        self.config = config
        action_space = self.config['trajectory']['args']['action_space']
        time_step = self.config['mujoco_sim_env']['args']["time_step"]
        if initial_qpos is None:
            initial_qpos = self.config['robot_config']['args']['initial_qpos']

        # TODO should read all configuration in config
        assert initial_qpos is not None, "Must initialize the initial q position of robot arm"
        n_actions = 7
        self.initial_qpos_value = np.array(list(initial_qpos.values())).copy()
        # # change time step in .xml file
        # tree = ET.parse(model_path)
        # root = tree.getroot()
        # for option in root.findall('option'):
        #     option.set("timestep", str(time_step))
        #
        # tree.write(model_path)

        super(TableTennisEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=n_actions,
            initial_qpos=initial_qpos)

        if action_space:
            self.action_space = spaces.Box(low=np.array([-2.6, -2.0, -2.8, -0.9, -4.8, -1.6, -2.2]),
                                           high=np.array([2.6, 2.0, 2.8, 3.1, 1.3, 1.6, 2.2]),
                                           dtype='float64')
        else:
            self.action_space = spaces.Box(low=np.array([-np.inf] * 7),
                                           high=np.array([-np.inf] * 7),
                                           dtype='float64')
        self.scale = None
        self.desired_pos = None
        self.n_actions = n_actions
        self.action = None
        self.time_step = time_step
        self.paddle_center_pos = self.sim.data.get_site_xpos('wam/paddle_center')
        if reward_obj is None:
            self.reward_obj = HierarchicalRewardTableTennis()
        else:
            self.reward_obj = reward_obj

        if initial_ball_state is not None:
            self.initial_ball_state = initial_ball_state
        else:
            self.initial_ball_state = ball_initialize(random=False)
        self.target_ball_pos = self.sim.data.get_site_xpos("target_ball")
        self.racket_center_pos = self.sim.data.get_site_xpos("wam/paddle_center")

    def close(self):
        if self.viewer is not None:
            glfw.destroy_window(self.viewer.window)
            # self.viewer.window.close()
            self.viewer = None
            self._viewers = {}

    # GoalEnv methods
    # ----------------------------
    def compute_reward(self, achieved_goal, goal, info):
        # reset the reward, if action valid
        # right_court_contact_obj = ["target_ball", "table_tennis_table_right_side"]
        # right_court_contact_detector = self.reward_obj.contact_detection(self, right_court_contact_obj)
        # if right_court_contact_detector:
        #     print("can detect the table ball contact")
        self.reward_obj.total_reward = 0
        # Stage 1 Hitting
        self.reward_obj.hitting(self)
        # if not hitted, return the highest reward
        if not self.reward_obj.goal_achievement:
            return self.reward_obj.highest_reward
        # # Stage 2 Right Table Contact
        # self.reward_obj.right_table_contact(self)
        # if not self.reward_obj.goal_achievement:
        #     return self.reward_obj.highest_reward
        # # Stage 2 Net Contact
        # self.reward_obj.net_contact(self)
        # if not self.reward_obj.goal_achievement:
        #     return self.reward_obj.highest_reward
        # Stage 3 Opponent court Contact
        # self.reward_obj.landing_on_opponent_court(self)
        # if not self.reward_obj.goal_achievement:
        # print("self.reward_obj.highest_reward: ", self.reward_obj.highest_reward)
        # TODO
        self.reward_obj.target_achievement(self)
        return self.reward_obj.highest_reward

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        [initial_x, initial_y, initial_z, v_x, v_y, v_z] = self.initial_ball_state
        self.sim.data.set_joint_qpos('tar:x', initial_x)
        self.sim.data.set_joint_qpos('tar:y', initial_y)
        self.sim.data.set_joint_qpos('tar:z', initial_z)
        self.energy_corrected = True
        self.give_reflection_reward = False

        # velocity is positive direction
        self.sim.data.set_joint_qvel('tar:x', v_x)
        self.sim.data.set_joint_qvel('tar:y', v_y)
        self.sim.data.set_joint_qvel('tar:z', v_z)

        # Apply gravity compensation
        if self.sim.data.qfrc_applied[:7] is not self.sim.data.qfrc_bias[:7]:
            self.sim.data.qfrc_applied[:7] = self.sim.data.qfrc_bias[:7]
        self.sim.forward()
        return True

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)

        # Apply gravity compensation
        if self.sim.data.qfrc_applied[:7] is not self.sim.data.qfrc_bias[:7]:
            self.sim.data.qfrc_applied[:7] = self.sim.data.qfrc_bias[:7]
        self.sim.forward()

        # Get the target position
        self.initial_paddle_center_xpos = self.sim.data.get_site_xpos('wam/paddle_center').copy()
        self.initial_paddle_center_vel = None  # self.sim.get_site_

    def _sample_goal(self):
        goal = self.initial_paddle_center_xpos[:3] + self.np_random.uniform(-0.2, 0.2, size=3)
        return goal.copy()

    def _get_obs(self):

        # positions of racket center
        paddle_center_pos = self.sim.data.get_site_xpos('wam/paddle_center')
        ball_pos = self.sim.data.get_site_xpos("target_ball")

        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        paddle_center_velp = self.sim.data.get_site_xvelp('wam/paddle_center') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

        wrist_state = robot_qpos[-3:]
        wrist_vel = robot_qvel[-3:] * dt  # change to a scalar if the gripper is made symmetric

        # achieved_goal = paddle_body_EE_pos
        obs = np.concatenate([
            paddle_center_pos, paddle_center_velp, wrist_state, wrist_vel
        ])

        out_dict = {
            'observation': obs.copy(),
            'achieved_goal': paddle_center_pos.copy(),
            'desired_goal': self.goal.copy(),
            'q_pos': self.sim.data.qpos[:].copy(),
            "ball_pos": ball_pos.copy(),
            # "hitting_flag": self.reward_obj.hitting_flag
        }

        return out_dict

    def _step_callback(self):
        pass

    def _set_action(self, action):
        # Apply gravity compensation
        if self.sim.data.qfrc_applied[:7] is not self.sim.data.qfrc_bias[:7]:
            self.sim.data.qfrc_applied[:7] = self.sim.data.qfrc_bias[:7]
        # print("set action process running")
        assert action.shape == (self.n_actions,)
        self.action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl = self.action[:]  # limit maximum change in position
        pos_ctrl = np.clip(pos_ctrl, self.action_space.low, self.action_space.high)

        # get desired trajectory
        self.sim.data.qpos[:7] = pos_ctrl
        self.sim.forward()
        self.desired_pos = self.sim.data.get_site_xpos('wam/paddle_center').copy()

        self.sim.data.ctrl[:] = pos_ctrl

    def _is_success(self, achieved_goal, desired_goal):
        pass


if __name__ == '__main__':
    render_mode = "human"  # "human" or "partial" or "final"
    env = TableTennisEnv()
    env.reset()
    # env.render(mode=render_mode)

    for i in range(200):
        # objective.load_result("/tmp/cma")
        # test with random actions
        ac = 2 * env.action_space.sample()
        # ac[0] += np.pi/2
        obs, rew, d, info = env.step(ac)
        env.render(mode=render_mode)

        print(rew)

        if d:
            break

    env.close()
