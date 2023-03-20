import numpy as np
from fancy_gym.envs.air_hockey.air_hockey import AirHockeyBase

from air_hockey_challenge.utils import robot_to_world
from air_hockey_challenge.framework import AirHockeyChallengeWrapper
from air_hockey_challenge.environments.planar import AirHockeyHit, AirHockeyDefend


class AirHockeyPlanarHit(AirHockeyBase):
    def __init__(self):
        super().__init__(env_id="3dof-hit", reward_function=self.planar_hit_reward)

    @staticmethod
    def planar_hit_reward(base_env: AirHockeyHit, obs, act, obs_, done):
        # init reward and env info
        rew = 0
        env_info = base_env.env_info

        # compute reward
        puck_pos, puck_vel = base_env.get_puck(obs_)  # puck_pos in world frame
        goal_pos = np.array([env_info["table"]["length"] / 2, 0, 0])  # goal_pos in world frame
        if done:
            # if puck in the opponent goal
            x_dis = puck_pos[0] - goal_pos[0]
            y_dis = np.abs(puck_pos[1]) - env_info["table"]["goal_width"] / 2
            if x_dis > 0 > y_dis:
                rew = 200
        else:
            if not base_env.has_hit:
                ee_pos, _ = base_env.get_ee()  # ee_pos in world frame
                ee_puck_dis = np.linalg.norm(ee_pos[:2] - puck_pos[:2])
                ee_puck_vec = (ee_pos[:2] - puck_pos[:2]) / ee_puck_dis
                puck_goal_dis = np.linalg.norm(puck_pos[:2] - goal_pos[:2])
                puck_goal_vec = (puck_pos[:2] - goal_pos[:2]) / puck_goal_dis
                cos_ang = np.clip(ee_puck_vec @ puck_goal_vec, 0, 1)
                rew = np.exp(-6 * (ee_puck_dis - 0.08)) * cos_ang
            else:
                rew_hit = 0.25 + min([1, (0.25 * puck_vel[0] ** 4)])
                rew_goal = 0
                if puck_pos[0] > 0.7:
                    sig = 0.1
                    rew_goal = 1. / (np.sqrt(2. * np.pi) * sig) * np.exp(-np.power((puck_pos[1] - 0) / sig, 2.) / 2)
                rew = 2 * rew_hit + 10 * rew_goal

        rew -= 1e-3 * np.linalg.norm(act)
        return rew


class AirHockeyPlanarDefend(AirHockeyBase):
    pass


def test_env(seed=0):
    env = AirHockeyPlanarHit()
    env.seed(seed)

    for i in range(5):
        obs = env.reset()
        stp = 0
        while True:
            act = env.action_space.sample()
            obs_, reward, done, info = env.step(act)
            ra = env.render("rgb_array")
            print(ra.shape)
            stp += 1
            if done:
                break


if __name__ == "__main__":
    test_env()
