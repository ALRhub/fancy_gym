from gym.envs.mujoco import mujoco_env
from gym import utils, spaces
import os
import numpy as np
from alr_envs.mujoco.ball_in_a_cup.ball_in_a_cup_reward import BallInACupReward
import mujoco_py


class ALRBallInACupEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, pd_control=True):
        self._steps = 0
        self.pd_control = pd_control

        self.xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets",
                                     "ball-in-a-cup_base" + ".xml")

        self.sim_time = 8  # seconds
        self.sim_steps = int(self.sim_time / (0.0005 * 4))  # circular dependency.. sim.dt <-> mujocoenv init <-> reward fct
        self.reward_function = BallInACupReward(self.sim_steps)

        self.start_pos = np.array([0.0, 0.58760536, 0.0, 1.36004913, 0.0, -0.32072943, -1.57])
        self.start_vel = np.zeros(7)
        # self.start_pos = np.array([0.58760536, 1.36004913, -0.32072943])
        self._q_pos = []
        self._q_vel = []
        # self.weight_matrix_scale = 50
        self.p_gains = 1*np.array([200, 300, 100, 100, 10, 10, 2.5])
        self.d_gains = 1*np.array([7, 15, 5, 2.5, 0.3, 0.3, 0.05])

        self.j_min = np.array([-2.6, -1.985, -2.8, -0.9, -4.55, -1.5707, -2.7])
        self.j_max = np.array([2.6, 1.985, 2.8, 3.14159, 1.25, 1.5707, 2.7])

        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, os.path.join(os.path.dirname(__file__), "assets", "ball-in-a-cup_base.xml"),
                                      frame_skip=4)

    @property
    def current_pos(self):
        return self.sim.data.qpos[0:7].copy()

    @property
    def current_vel(self):
        return self.sim.data.qvel[0:7].copy()

    def reset_model(self):
        init_pos_all = self.init_qpos.copy()
        init_pos_robot = self.start_pos
        init_vel = np.zeros_like(init_pos_all)
        ball_id = self.sim.model._body_name2id["ball"]
        goal_final_id = self.sim.model._site_name2id["cup_goal_final"]
        # self.set_state(start_pos, start_vel)
        self._steps = 0
        self.reward_function.reset()
        self._q_pos = []
        self._q_vel = []

        # Reset the system
        self.sim.data.qpos[:] = init_pos_all
        self.sim.data.qvel[:] = init_vel
        self.sim.data.qpos[0:7] = init_pos_robot

        self.sim.step()

        self.sim.data.qpos[:] = init_pos_all
        self.sim.data.qvel[:] = init_vel
        self.sim.data.qpos[0:7] = init_pos_robot
        self.sim.data.body_xpos[ball_id, :] = np.copy(self.sim.data.site_xpos[goal_final_id, :]) - np.array([0., 0., 0.329])

        # Stabilize the system around the initial position
        for i in range(0, 2000):
            self.sim.data.qpos[7:] = 0.
            self.sim.data.qvel[7:] = 0.
            # self.sim.data.qpos[7] = -0.2
            cur_pos = self.sim.data.qpos[0:7].copy()
            cur_vel = self.sim.data.qvel[0:7].copy()
            trq = self.p_gains * (init_pos_robot - cur_pos) + self.d_gains * (np.zeros_like(init_pos_robot) - cur_vel)
            self.sim.data.qfrc_applied[0:7] = trq + self.sim.data.qfrc_bias[:7].copy()
            self.sim.step()
            # self.render()

        for i in range(0, 2000):
            cur_pos = self.sim.data.qpos[0:7].copy()
            cur_vel = self.sim.data.qvel[0:7].copy()
            trq = self.p_gains * (init_pos_robot - cur_pos) + self.d_gains * (np.zeros_like(init_pos_robot) - cur_vel)
            self.sim.data.qfrc_applied[0:7] = trq + self.sim.data.qfrc_bias[:7].copy()
            self.sim.step()
            # self.render()

    def do_simulation(self, ctrl, n_frames):
        # cur_pos = self.sim.data.qpos[0:7].copy()
        # cur_vel = self.sim.data.qvel[0:7].copy()
        # des_pos = ctrl[:7]
        # des_vel = ctrl[7:]
        # trq = self.p_gains * (des_pos - cur_pos) + self.d_gains * (des_vel - cur_vel)
        if self.pd_control:
            self.sim.data.qfrc_applied[0:7] = ctrl + self.sim.data.qfrc_bias[:7].copy()
        else:
            self.sim.data.ctrl[:] = ctrl

        for _ in range(n_frames):
            try:
                self.sim.step()
            except mujoco_py.builder.MujocoException as e:
                print("Error in simulation: " + str(e))
                # error = True
                # Copy the current torque as if it would have been applied until the end of the trajectory
                # for i in range(k + 1, sim_time):
                #     torques.append(trq)
                return True

        return False

    def step(self, a):
        # Apply gravity compensation
        # if not np.all(self.sim.data.qfrc_applied[:7] == self.sim.data.qfrc_bias[:7]):
        #     self.sim.data.qfrc_applied[:7] = self.sim.data.qfrc_bias[:7]

        reward_dist = 0.0
        angular_vel = 0.0
        # if self._steps >= self.steps_before_reward:
        #     vec = self.get_body_com("fingertip") - self.get_body_com("target")
        #     reward_dist -= self.reward_weight * np.linalg.norm(vec)
        #     angular_vel -= np.linalg.norm(self.sim.data.qvel.flat[:self.n_links])
        reward_ctrl = - np.square(a).sum()
        # reward_balance = - self.balance_weight * np.abs(
        #     angle_normalize(np.sum(self.sim.data.qpos.flat[:self.n_links]), type="rad"))
        #
        # reward = reward_dist + reward_ctrl + angular_vel + reward_balance
        # self.do_simulation(a, self.frame_skip)

        joint_cons_viol = self.check_traj_in_joint_limits()

        crash = self.do_simulation(a, self.frame_skip)

        self._q_pos.append(self.sim.data.qpos[0:7].ravel().copy())
        self._q_vel.append(self.sim.data.qvel[0:7].ravel().copy())

        ob = self._get_obs()

        if not crash and not joint_cons_viol:
            reward, success, collision = self.reward_function.compute_reward(a, self.sim, self._steps)
            done = success or self._steps == self.sim_steps - 1 or collision
            self._steps += 1
        else:
            reward = -1000
            done = True
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl,
                                      velocity=angular_vel, # reward_balance=reward_balance,
                                      # end_effector=self.get_body_com("fingertip").copy(),
                                      goal=self.goal if hasattr(self, "goal") else None,
                                      traj=self._q_pos)

    def check_traj_in_joint_limits(self):
        return any(self.current_pos > self.j_max) or any(self.current_pos < self.j_min)

    def check_collision(self):
        for coni in range(0, sim.data.ncon):
            con = sim.data.contact[coni]

            collision = con.geom1 in self.collision_ids and con.geom2 == self.ball_collision_id
            collision_trans = con.geom1 == self.ball_collision_id and con.geom2 in self.collision_ids

            if collision or collision_trans:
                return True
        return False

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:7]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            # self.get_body_com("target"),  # only return target to make problem harder
            [self._steps],
        ])

    def extend_des_pos(self, des_pos):
        des_pos_full = self.start_pos.copy()
        des_pos_full[1] = des_pos[0]
        des_pos_full[3] = des_pos[1]
        des_pos_full[5] = des_pos[2]
        return des_pos_full

    def extend_des_vel(self, des_vel):
        des_vel_full = self.start_vel.copy()
        des_vel_full[1] = des_vel[0]
        des_vel_full[3] = des_vel[1]
        des_vel_full[5] = des_vel[2]
        return des_vel_full

if __name__ == "__main__":
    env = ALRBallInACupEnv()
    env.reset()
    env.render()
    for i in range(4000):
        # objective.load_result("/tmp/cma")
        # test with random actions
        # ac = 0.1 * env.action_space.sample()
        # ac = -np.array([i, i, i]) / 10000 + np.array([env.start_pos[1], env.start_pos[3], env.start_pos[5]])
        ac = np.array([0., -0.1, 0, 0, 0, 0, 0])
        # ac[0] += np.pi/2
        obs, rew, d, info = env.step(ac)
        env.render()

        print(rew)

        if d:
            break

    env.close()

