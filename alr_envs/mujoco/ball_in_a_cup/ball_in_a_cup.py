from gym.envs.mujoco import mujoco_env
from gym import utils
import os
import numpy as np
from alr_envs.mujoco.ball_in_a_cup.ball_in_a_cup_reward import BallInACupReward
import mujoco_py


class ALRBallInACupEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, ):
        self._steps = 0

        self.xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets",
                                     "ball-in-a-cup_base" + ".xml")

        self.sim_time = 8  # seconds
        self.sim_steps = int(self.sim_time / (0.0005 * 4))  # circular dependency.. sim.dt <-> mujocoenv init <-> reward fct
        self.reward_function = BallInACupReward(self.sim_steps)

        self.start_pos = np.array([0.0, 0.58760536, 0.0, 1.36004913, 0.0, -0.32072943, -1.57])
        self._q_pos = []

        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, os.path.join(os.path.dirname(__file__), "assets", "ball-in-a-cup_base.xml"),
                                      frame_skip=4)

    def reset_model(self):
        start_pos = self.init_qpos.copy()
        start_pos[0:7] = self.start_pos
        start_vel = np.zeros_like(start_pos)
        self.set_state(start_pos, start_vel)
        self._steps = 0
        self.reward_function.reset()
        self._q_pos = []

    def do_simulation(self, ctrl, n_frames):
        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            try:
                self.sim.step()
            except mujoco_py.builder.MujocoException as e:
                # print("Error in simulation: " + str(e))
                # error = True
                # Copy the current torque as if it would have been applied until the end of the trajectory
                # for i in range(k + 1, sim_time):
                #     torques.append(trq)
                return True

        return False

    def step(self, a):
        # Apply gravity compensation
        if not np.all(self.sim.data.qfrc_applied[:7] == self.sim.data.qfrc_bias[:7]):
            self.sim.data.qfrc_applied[:7] = self.sim.data.qfrc_bias[:7]

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

        crash = self.do_simulation(a, self.frame_skip)

        self._q_pos.append(self.sim.data.qpos[0:7].ravel().copy())

        ob = self._get_obs()

        if not crash:
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

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:7]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            # self.get_body_com("target"),  # only return target to make problem harder
            [self._steps],
        ])



if __name__ == "__main__":
    env = ALRBallInACupEnv()
    env.reset()
    for i in range(2000):
        # objective.load_result("/tmp/cma")
        # test with random actions
        # ac = 0.0 * env.action_space.sample()
        ac = env.start_pos
        # ac[0] += np.pi/2
        obs, rew, d, info = env.step(ac)
        env.render()

        print(rew)

        if d:
            break

    env.close()

