import os

import gym.envs.mujoco
import gym.envs.mujoco as mujoco_env
import mujoco_py.builder
import numpy as np
from gym import utils

from mp_env_api.mp_wrappers.detpmp_wrapper import DetPMPWrapper
from mp_env_api.utils.policies import PDControllerExtend


def make_detpmp_env(**kwargs):
    name = kwargs.pop("name")
    _env = gym.make(name)
    policy = PDControllerExtend(_env, p_gains=kwargs.pop('p_gains'), d_gains=kwargs.pop('d_gains'))
    kwargs['policy_type'] = policy
    return DetPMPWrapper(_env, **kwargs)


class ALRBallInACupPDEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, frame_skip=4, apply_gravity_comp=True, simplified: bool = False,
                 reward_type: str = None, context: np.ndarray = None):
        utils.EzPickle.__init__(**locals())
        self._steps = 0

        self.xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "biac_base.xml")

        self.max_ctrl = np.array([150., 125., 40., 60., 5., 5., 2.])

        self.j_min = np.array([-2.6, -1.985, -2.8, -0.9, -4.55, -1.5707, -2.7])
        self.j_max = np.array([2.6, 1.985, 2.8, 3.14159, 1.25, 1.5707, 2.7])

        self.context = context
        self.apply_gravity_comp = apply_gravity_comp
        self.simplified = simplified

        self._start_pos = np.array([0.0, 0.58760536, 0.0, 1.36004913, 0.0, -0.32072943, -1.57])
        self._start_vel = np.zeros(7)

        self.sim_time = 8  # seconds
        self._dt = 0.02
        self.ep_length = 4000  # based on 8 seconds with dt = 0.02 int(self.sim_time / self.dt)
        if reward_type == "no_context":
            from alr_envs.mujoco.ball_in_a_cup.ball_in_a_cup_reward_simple import BallInACupReward
            reward_function = BallInACupReward
        elif reward_type == "contextual_goal":
            from alr_envs.mujoco.ball_in_a_cup.ball_in_a_cup_reward import BallInACupReward
            reward_function = BallInACupReward
        else:
            raise ValueError("Unknown reward type: {}".format(reward_type))
        self.reward_function = reward_function(self)

        mujoco_env.MujocoEnv.__init__(self, self.xml_path, frame_skip)

    @property
    def dt(self):
        return self._dt

    # TODO: @Max is this even needed?
    @property
    def start_vel(self):
        if self.simplified:
            return self._start_vel[1::2]
        else:
            return self._start_vel

    # def _set_action_space(self):
    #     if self.simplified:
    #         bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)[1::2]
    #     else:
    #         bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
    #     low, high = bounds.T
    #     self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
    #     return self.action_space

    def reset(self):
        self.reward_function.reset(None)
        return super().reset()

    def reset_model(self):
        init_pos_all = self.init_qpos.copy()
        init_pos_robot = self._start_pos
        init_vel = np.zeros_like(init_pos_all)

        self._steps = 0
        self._q_pos = []
        self._q_vel = []

        start_pos = init_pos_all
        start_pos[0:7] = init_pos_robot

        self.set_state(start_pos, init_vel)

        return self._get_obs()

    def step(self, a):
        reward_dist = 0.0
        angular_vel = 0.0
        reward_ctrl = - np.square(a).sum()

        # if self.simplified:
        #     tmp = np.zeros(7)
        #     tmp[1::2] = a
        #     a = tmp

        if self.apply_gravity_comp:
            a += self.sim.data.qfrc_bias[:len(a)].copy() / self.model.actuator_gear[:, 0]

        crash = False
        try:
            self.do_simulation(a, self.frame_skip)
        except mujoco_py.builder.MujocoException:
            crash = True
        # joint_cons_viol = self.check_traj_in_joint_limits()

        ob = self._get_obs()

        if not crash:
            reward, success, is_collided = self.reward_function.compute_reward(a)
            done = success or is_collided  # self._steps == self.sim_steps - 1
            self._steps += 1
        else:
            reward = -2000
            success = False
            is_collided = False
            done = True

        return ob, reward, done, dict(reward_dist=reward_dist,
                                      reward_ctrl=reward_ctrl,
                                      velocity=angular_vel,
                                      # traj=self._q_pos,
                                      action=a,
                                      q_pos=self.sim.data.qpos[0:7].ravel().copy(),
                                      q_vel=self.sim.data.qvel[0:7].ravel().copy(),
                                      is_success=success,
                                      is_collided=is_collided, sim_crash=crash)

    def check_traj_in_joint_limits(self):
        return any(self.current_pos > self.j_max) or any(self.current_pos < self.j_min)

    # TODO: extend observation space
    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:7]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            # self.get_body_com("target"),  # only return target to make problem harder
            [self._steps],
        ])

    # These functions are for the task with 3 joint actuations
    def extend_des_pos(self, des_pos):
        des_pos_full = self._start_pos.copy()
        des_pos_full[1] = des_pos[0]
        des_pos_full[3] = des_pos[1]
        des_pos_full[5] = des_pos[2]
        return des_pos_full

    def extend_des_vel(self, des_vel):
        des_vel_full = self._start_vel.copy()
        des_vel_full[1] = des_vel[0]
        des_vel_full[3] = des_vel[1]
        des_vel_full[5] = des_vel[2]
        return des_vel_full

    def render(self, render_mode, **render_kwargs):
        if render_mode == "plot_trajectory":
            if self._steps == 1:
                import matplotlib.pyplot as plt
                # plt.ion()
                self.fig, self.axs = plt.subplots(3, 1)

            if self._steps <= 1750:
                for ax, cp in zip(self.axs, self.current_pos[1::2]):
                    ax.scatter(self._steps, cp, s=2, marker=".")

            # self.fig.show()

        else:
            super().render(render_mode, **render_kwargs)


if __name__ == "__main__":
    env = ALRBallInACupPDEnv(reward_type="no_context", simplified=True)
    # env = gym.make("alr_envs:ALRBallInACupPDSimpleDetPMP-v0")
    # ctxt = np.array([-0.20869846, -0.66376693, 1.18088501])

    # env.configure(ctxt)
    env.reset()
    env.render("human")
    for i in range(16000):
        # test with random actions
        ac = 0.02 * env.action_space.sample()[0:7]
        # ac = env.start_pos
        # ac[0] += np.pi/2
        obs, rew, d, info = env.step(ac)
        env.render("human")

        print(rew)

        if d:
            break

    env.close()
