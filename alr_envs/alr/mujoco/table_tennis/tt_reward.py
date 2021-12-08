import numpy as np
from alr_envs.alr.mujoco.table_tennis.tt_gym import MAX_EPISODE_STEPS


class TT_Reward:

    def __init__(self, ctxt_dim):
        self.ctxt_dim = ctxt_dim
        self.c_goal = None          # current desired landing point
        self.c_ball_traj = []
        self.c_racket_traj = []
        self.constant = 8
        self.ball_landing_pos = None
        self.hit_ball = False
        self.ball_contact_after_hit = False
        self.actions = []

    def get_reward(self, env, action):
        self.actions.append(action)
        done = False
        episode_end = False if env.time_steps + 1 < MAX_EPISODE_STEPS else True
        if not self.hit_ball:
            self.hit_ball = env._contact_checker(env.ball_contact_id, env.paddle_contact_id_1) # check for one side
            if not self.hit_ball:
                self.hit_ball = env._contact_checker(env.ball_contact_id, env.paddle_contact_id_2) # check for other side
        if self.hit_ball:
            if not self.ball_contact_after_hit:
                if env._contact_checker(env.ball_contact_id, env.floor_contact_id):  # first check contact with floor
                    self.ball_contact_after_hit = True
                    self.ball_landing_pos = env.sim.data.body_xpos[env.ball_id]
                elif env._contact_checker(env.ball_contact_id, env.table_contact_id): # second check contact with table
                    self.ball_contact_after_hit = True
                    self.ball_landing_pos = env.sim.data.body_xpos[env.ball_id]
        c_ball_pos = env.sim.data.body_xpos[env.ball_id]
        racket_pos = env.sim.data.geom_xpos[env.racket_id]        # TODO: use this to reach out the position of the paddle?

        self.c_ball_traj.append(c_ball_pos.copy())
        self.c_racket_traj.append(racket_pos.copy())
        if self.ball_landing_pos is not None:
            done = True
            episode_end =True

        if not episode_end:
            return 0, done
        else:
            # # seems to work for episodic case
            min_r_b_dist = np.min(np.linalg.norm(np.array(self.c_ball_traj) - np.array(self.c_racket_traj), axis=1))
            if not self.hit_ball:
                return 0.2 * (1 - np.tanh(min_r_b_dist**2)), done
            else:
                if self.ball_landing_pos is None:
                    min_b_des_b_dist = np.min(np.linalg.norm(np.array(self.c_ball_traj) - self.c_goal, axis=1))
                    return 2 * (1 - np.tanh(min_r_b_dist ** 2)) + (1 - np.tanh(min_b_des_b_dist**2)), done
                else:
                    min_b_des_b_land_dist = np.linalg.norm(self.c_goal[:2] - self.ball_landing_pos[:2])
                    over_net_bonus = int(self.ball_landing_pos[0] < 0)
                    return 2 * (1 - np.tanh(min_r_b_dist ** 2)) + 4 * (1 - np.tanh(min_b_des_b_land_dist ** 2)) + over_net_bonus - 1e-3 * np.sum(np.square(self.actions)), done


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
        self.ball_landing_pos = None
        self.hit_ball = False
        self.ball_contact_after_hit = False
        self.actions = []
