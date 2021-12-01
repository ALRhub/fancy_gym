import numpy as np


class TT_Reward:

    def __init__(self, ctxt_dim):
        self.ctxt_dim = ctxt_dim
        self.c_goal = None          # current desired landing point
        self.c_ball_traj = []
        self.c_racket_traj = []
        self.constant = 8

    def get_reward(self, episode_end, ball_position, racket_pos, hited_ball, ball_landing_pos):
        self.c_ball_traj.append(ball_position.copy())
        self.c_racket_traj.append(racket_pos.copy())
        if not episode_end:
            return 0
        else:
            # # seems to work for episodic case
            min_r_b_dist = np.min(np.linalg.norm(np.array(self.c_ball_traj) - np.array(self.c_racket_traj), axis=1))
            if not hited_ball:
                return 0.2 * (1- np.tanh(min_r_b_dist**2))
            else:
                if ball_landing_pos is None:
                    min_b_des_b_dist = np.min(np.linalg.norm(np.array(self.c_ball_traj)[:,:2] - self.c_goal[:2], axis=1))
                    return 2 * (1 - np.tanh(min_r_b_dist ** 2)) + (1 - np.tanh(min_b_des_b_dist**2))
                else:
                    min_b_des_b_land_dist = np.linalg.norm(self.c_goal[:2] - ball_landing_pos[:2])
                    over_net_bonus = int(ball_landing_pos[0] < 0)
                    return 2 * (1 - np.tanh(min_r_b_dist ** 2)) + 4 * (1 - np.tanh(min_b_des_b_land_dist ** 2)) + over_net_bonus


            # if not hited_ball:
            #     min_r_b_dist = 1 + np.min(np.linalg.norm(np.array(self.c_ball_traj) - np.array(self.c_racket_traj), axis=1))
            #     return -min_r_b_dist
            # else:
            #     if ball_landing_pos is None:
            #         dist_to_des_pos = 1-np.power(np.linalg.norm(self.c_goal - ball_position), 0.75)/self.constant
            #     else:
            #         dist_to_des_pos = 1-np.power(np.linalg.norm(self.c_goal - ball_landing_pos), 0.75)/self.constant
            #     if dist_to_des_pos < -0.2:
            #         dist_to_des_pos = -0.2
            #     return -dist_to_des_pos

    def reset(self, goal):
        self.c_goal = goal.copy()
        self.c_ball_traj = []
        self.c_racket_traj = []