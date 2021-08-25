import numpy as np
import logging


class HierarchicalRewardTableTennis(object):
    """Class for hierarchical reward function for table tennis experiment.

    Return Highest Reward.
    Reward = 0
    Step 1: Action Valid. Upper Bound 0
                [-∞, 0]
                Reward += -1 * |hit_duration - hit_duration_threshold| * |hit_duration < hit_duration_threshold| * 10
    Step 2: Hitting. Upper Bound 2
                if hitting:
                    [0, 2]
                    Reward = 2 * (1 - tanh(|shortest_hitting_dist|))
                if not hitting:
                    [0, 0.2]
                    Reward = 2 * (1 - tanh(|shortest_hitting_dist|))
    Step 3: Target Point Achievement. Upper Bound 6
                [0, 4]
                if table_contact_detector:
                    Reward += 1
                    Reward += (1 - tanh(|shortest_hitting_dist|)) * 2
                    if contact_coordinate[0] < 0:
                        Reward += 1
                    else:
                        Reward += 0
                elif:
                    Reward += (1 - tanh(|shortest_hitting_dist|))
    """

    def __init__(self):
        self.reward = None
        self.goal_achievement = False
        self.total_reward = 0
        self.shortest_hitting_dist = 1000
        self.highest_reward = -1000
        self.lowest_corner_dist = 100
        self.right_court_contact_detector = False
        self.table_contact_detector = False
        self.floor_contact_detector = False
        self.radius = 0.025
        self.min_ball_x_pos = 100
        self.hit_contact_detector = False
        self.net_contact_detector = False
        self.ratio = 1
        self.lowest_z = 100
        self.target_flag = False
        self.dist_target_virtual = 100
        self.ball_z_pos_lowest = 100
        self.hitting_flag = False
        self.hitting_time_point = None
        self.ctxt_dim = None
        self.context_range_bounds = None
        # self.ctxt_out_of_range_punishment = None
        # self.ctxt_in_side_of_range_punishment = None
    #
    # def check_where_invalid(self, ctxt, context_range_bounds, set_to_valid_region=False):
    #     idx_max = []
    #     idx_min = []
    #     for dim in range(self.ctxt_dim):
    #         min_dim = context_range_bounds[0][dim]
    #         max_dim = context_range_bounds[1][dim]
    #         idx_max_c = np.where(ctxt[:, dim] > max_dim)[0]
    #         idx_min_c = np.where(ctxt[:, dim] < min_dim)[0]
    #         if set_to_valid_region:
    #             if idx_max_c.shape[0] != 0:
    #                 ctxt[idx_max_c, dim] = max_dim
    #             if idx_min_c.shape[0] != 0:
    #                 ctxt[idx_min_c, dim] = min_dim
    #         idx_max.append(idx_max_c)
    #         idx_min.append(idx_min_c)
    #     return idx_max, idx_min, ctxt

    def check_valid(self, scale, context_range_bounds):

        min_dim = context_range_bounds[0][0]
        max_dim = context_range_bounds[1][0]
        valid = (scale < max_dim) and (scale > min_dim)
        return valid

    @classmethod
    def goal_distance(cls, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def refresh_highest_reward(self):
        if self.total_reward >= self.highest_reward:
            self.highest_reward = self.total_reward

    def duration_valid(self):
        pass

    def huge_value_unstable(self):
        self.total_reward += -10
        self.highest_reward = -1

    def context_valid(self, context):
        valid = self.check_valid(context.copy(), context_range_bounds=self.context_range_bounds)
        # when using dirac punishments
        if valid:
            self.total_reward += 1 # If Action Valid and Context Valid, total_reward = 0
        else:
            self.total_reward += 0
        self.refresh_highest_reward()



        # If in the ctxt, add 1, otherwise, 0

    def action_valid(self, durations=None):
        """Ensure the execution of the robot movement with parameters which are in a valid domain.

        Time should always be positive,
        the joint position of the robot should be a subset of [−π, π].
        if all parameters are valid, the robot gets a zero score,
        otherwise it gets a negative score proportional to how much it is beyond the valid parameter domain.

        Returns:
            rewards: if valid, reward is equal to 0.
            if not valid, reward is negative and proportional to the distance beyond the valid parameter domain
        """
        assert durations.shape[0] == 2, "durations type should be np.array and the shape should be 2"
        # pre_duration = durations[0]
        hit_duration = durations[1]
        # pre_duration_thres = 0.01
        hit_duration_thres = 1
        # self.goal_achievement = np.all(
        #     [(pre_duration > pre_duration_thres), (hit_duration > hit_duration_thres), (0.3 < pre_duration < 0.6)])
        self.goal_achievement = (hit_duration > hit_duration_thres)
        if self.goal_achievement:
            self.total_reward = -1
            self.goal_achievement = True
        else:
            # self.total_reward += -1 * ((np.abs(pre_duration - pre_duration_thres) * int(
            #     pre_duration < pre_duration_thres) + np.abs(hit_duration - hit_duration_thres) * int(
            #     hit_duration < hit_duration_thres)) * 10)
            self.total_reward = -1 * ((np.abs(hit_duration - hit_duration_thres) * int(
                hit_duration < hit_duration_thres)) * 10)
            self.total_reward += -1
            self.goal_achievement = False
        self.refresh_highest_reward()

    def motion_penalty(self, action, high_motion_penalty):
        """Protects the robot from high acceleration and dangerous movement.
        """
        if not high_motion_penalty:
            reward_ctrl = - 0.05 * np.square(action).sum()
        else:
            reward_ctrl = - 0.075 * np.square(action).sum()
        self.total_reward += reward_ctrl
        self.refresh_highest_reward()
        self.goal_achievement = True

    def hitting(self, env):  # , target_ball_pos, racket_center_pos, hit_contact_detector=False
        """Hitting reward calculation

        If racket successfully hit the ball, the reward +1
        Otherwise calculate the distance between the center of racket and the center of ball,
        reward = tanh(r/dist) if dist<1 reward almost 2 , if dist >= 1 reward is between [0, 0.2]


        Args:
            env:

        Returns:

        """

        hit_contact_obj = ["target_ball", "bat"]
        target_ball_pos = env.target_ball_pos
        racket_center_pos = env.racket_center_pos
        # hit contact detection
        # Record the hitting history
        self.hitting_flag = False
        if not self.hit_contact_detector:
            self.hit_contact_detector = self.contact_detection(env, hit_contact_obj)
            if self.hit_contact_detector:
                print("First time detect hitting")
                self.hitting_flag = True
        if self.hit_contact_detector:

            # TODO
            dist = self.goal_distance(target_ball_pos, racket_center_pos)
            if dist < 0:
                dist = 0
            # print("goal distance is:", dist)
            if dist <= self.shortest_hitting_dist:
                self.shortest_hitting_dist = dist
            # print("shortest_hitting_dist is:", self.shortest_hitting_dist)
            # Keep the shortest hitting distance.
            dist_reward = 2 * (1 - np.tanh(np.abs(self.shortest_hitting_dist)))

            # TODO sparse
            # dist_reward = 2

            self.total_reward += dist_reward
            self.goal_achievement = True

            # if self.hitting_time_point is not None and self.hitting_time_point > 600:
            #     self.total_reward += 1

        else:
            dist = self.goal_distance(target_ball_pos, racket_center_pos)
            if dist <= self.shortest_hitting_dist:
                self.shortest_hitting_dist = dist
            dist_reward = 1 - np.tanh(self.shortest_hitting_dist)
            reward = 0.2 * dist_reward  # because it does not hit the ball, so multiply 0.2
            self.total_reward += reward
            self.goal_achievement = False

        self.refresh_highest_reward()

    @classmethod
    def relu(cls, x):
        return np.maximum(0, x)

    # def right_table_contact(self, env):
    #     right_court_contact_obj = ["target_ball", "table_tennis_table_right_side"]
    #     if env.target_ball_pos[0] >= 0 and env.target_ball_pos[2] >= 0.7:
    #         # update right court contact detection
    #         if not self.right_court_contact_detector:
    #             self.right_court_contact_detector = self.contact_detection(env, right_court_contact_obj)
    #             if self.right_court_contact_detector:
    #                 self.contact_x_pos = env.target_ball_pos[0]
    #         if self.right_court_contact_detector:
    #             self.total_reward += 1 - norm(0.685, 1).pdf(self.contact_x_pos)  # x axis middle of right table
    #             self.goal_achievement = False
    #         else:
    #             self.total_reward += 1
    #             self.goal_achievement = True
    #     # else:
    #     #     self.total_reward += 0
    #     #     self.goal_achievement = False
    #     self.refresh_highest_reward()

    # def net_contact(self, env):
    #     net_contact_obj = ["target_ball", "table_tennis_net"]
    #     # net_contact_detector = self.contact_detection(env, net_contact_obj)
    #     # ball_x_pos = env.target_ball_pos[0]
    #     # if self.min_ball_x_pos >= ball_x_pos:
    #     #     self.min_ball_x_pos = ball_x_pos
    #     # table_left_edge_x_pos = -1.37
    #     # if np.abs(ball_x_pos) <= 0.01:  # x threshold of net
    #     #     if self.lowest_z >= env.target_ball_pos[2]:
    #     #         self.lowest_z = env.target_ball_pos[2]
    #     #     # construct a gaussian distribution of z
    #     #     z_reward = 4 - norm(0, 0.1).pdf(self.lowest_z - 0.07625)  # maximum 4
    #     #     self.total_reward += z_reward
    #     # self.total_reward += 2 - np.minimum(1, self.relu(np.abs(self.min_ball_x_pos)))
    #     if not self.net_contact_detector:
    #         self.net_contact_detector = self.contact_detection(env, net_contact_obj)
    #     if self.net_contact_detector:
    #         self.total_reward += 0  # very high cost
    #         self.goal_achievement = False
    #     else:
    #         self.total_reward += 1
    #         self.goal_achievement = True
    #     self.refresh_highest_reward()

    # def landing_on_opponent_court(self, env):
    #     # Very sparse reward
    #     # don't contact the right side court
    #     # right_court_contact_obj = ["target_ball", "table_tennis_table_right_side"]
    #     # right_court_contact_detector = self.contact_detection(env, right_court_contact_obj)
    #     left_court_contact_obj = ["target_ball", "table_tennis_table_left_side"]
    #     # left_court_contact_detector = self.contact_detection(env, left_court_contact_obj)
    #     # record the contact history
    #     # if not self.right_court_contact_detector:
    #     #     self.right_court_contact_detector = self.contact_detection(env, right_court_contact_obj)
    #     if not self.table_contact_detector:
    #         self.table_contact_detector = self.contact_detection(env, left_court_contact_obj)
    #
    #     dist_left_up_corner = self.goal_distance(env.target_ball_pos, env.sim.data.get_site_xpos("left_up_corner"))
    #     dist_middle_up_corner = self.goal_distance(env.target_ball_pos, env.sim.data.get_site_xpos("middle_up_corner"))
    #     dist_left_down_corner = self.goal_distance(env.target_ball_pos, env.sim.data.get_site_xpos("left_down_corner"))
    #     dist_middle_down_corner = self.goal_distance(env.target_ball_pos,
    #                                                  env.sim.data.get_site_xpos("middle_down_corner"))
    #     dist_array = np.array(
    #         [dist_left_up_corner, dist_middle_up_corner, dist_left_down_corner, dist_middle_down_corner])
    #     dist_corner = np.amin(dist_array)
    #     if self.lowest_corner_dist >= dist_corner:
    #         self.lowest_corner_dist = dist_corner
    #
    #     right_contact_cost = 1
    #     left_contact_reward = 2
    #     dist_left_table_reward = (2 - np.tanh(self.lowest_corner_dist))
    #     # TODO Try multi dimensional gaussian distribution
    #     # contact only the left side court
    #     if self.right_court_contact_detector:
    #         self.total_reward += 0
    #         self.goal_achievement = False
    #         if self.table_contact_detector:
    #             self.total_reward += left_contact_reward
    #             self.goal_achievement = False
    #         else:
    #             self.total_reward += dist_left_table_reward
    #             self.goal_achievement = False
    #     else:
    #         self.total_reward += right_contact_cost
    #         if self.table_contact_detector:
    #             self.total_reward += left_contact_reward
    #             self.goal_achievement = True
    #         else:
    #             self.total_reward += dist_left_table_reward
    #             self.goal_achievement = False
    #     self.refresh_highest_reward()
    #     # if self.left_court_contact_detector and not self.right_court_contact_detector:
    #     #     self.total_reward += self.ratio * left_contact_reward
    #     #     print("only left court reward return!!!!!!!!!")
    #     #     print("contact only left court!!!!!!")
    #     #     self.goal_achievement = True
    #     # # no contact with table
    #     # elif not self.right_court_contact_detector and not self.left_court_contact_detector:
    #     #     self.total_reward += 0 + self.ratio * dist_left_table_reward
    #     #     self.goal_achievement = False
    #     # # contact both side
    #     # elif self.right_court_contact_detector and self.left_court_contact_detector:
    #     #     self.total_reward += self.ratio * (left_contact_reward - right_contact_cost)  # cost of contact of right court
    #     #     self.goal_achievement = False
    #     # # contact only the right side court
    #     # elif self.right_court_contact_detector and not self.left_court_contact_detector:
    #     #     self.total_reward += 0 + self.ratio * (
    #     #                 dist_left_table_reward - right_contact_cost)  # cost of contact of right court
    #     #     self.goal_achievement = False

    def target_achievement(self, env):
        target_coordinate = np.array([-0.5, -0.5])
        # net_contact_obj = ["target_ball", "table_tennis_net"]
        table_contact_obj = ["target_ball", "table_tennis_table"]
        floor_contact_obj = ["target_ball", "floor"]

        if 0.78 < env.target_ball_pos[2] < 0.8:
            dist_target_virtual = np.linalg.norm(env.target_ball_pos[:2] - target_coordinate)
            if self.dist_target_virtual > dist_target_virtual:
                self.dist_target_virtual = dist_target_virtual
        if -0.07 < env.target_ball_pos[0] < 0.07 and env.sim.data.get_joint_qvel('tar:x') < 0:
            if self.ball_z_pos_lowest > env.target_ball_pos[2]:
                self.ball_z_pos_lowest = env.target_ball_pos[2].copy()
        # if not self.net_contact_detector:
        #     self.net_contact_detector = self.contact_detection(env, net_contact_obj)
        if not self.table_contact_detector:
            self.table_contact_detector = self.contact_detection(env, table_contact_obj)
        if not self.floor_contact_detector:
            self.floor_contact_detector = self.contact_detection(env, floor_contact_obj)
        if not self.target_flag:
            # Table Contact Reward.
            if self.table_contact_detector:
                self.total_reward += 1
                # only update when the first contact because of the flag
                contact_coordinate = env.target_ball_pos[:2].copy()
                print("contact table ball coordinate: ", env.target_ball_pos)
                logging.info("contact table ball coordinate: {}".format(env.target_ball_pos))
                dist_target = np.linalg.norm(contact_coordinate - target_coordinate)
                self.total_reward += (1 - np.tanh(dist_target)) * 2
                self.target_flag = True
                # Net Contact Reward. Precondition: Table Contact exits.
                if contact_coordinate[0] < 0:
                    print("left table contact")
                    logging.info("~~~~~~~~~~~~~~~left table contact~~~~~~~~~~~~~~~")
                    self.total_reward += 1
                    # TODO Z coordinate reward
                    # self.total_reward += np.maximum(np.tanh(self.ball_z_pos_lowest), 0)
                    self.goal_achievement = True
                else:
                    print("right table contact")
                    logging.info("~~~~~~~~~~~~~~~right table contact~~~~~~~~~~~~~~~")
                    self.total_reward += 0
                    self.goal_achievement = False
                # if self.net_contact_detector:
                #     self.total_reward += 0
                #     self.goal_achievement = False
                # else:
                #     self.total_reward += 1
                #     self.goal_achievement = False
            # Floor Contact Reward. Precondition: Table Contact exits.
            elif self.floor_contact_detector:
                self.total_reward += (1 - np.tanh(self.dist_target_virtual))
                self.target_flag = True
                self.goal_achievement = False
            # No Contact of Floor or Table, flying
            else:
                pass
        # else:
        # print("Flag is True already")
        self.refresh_highest_reward()

    def distance_to_target(self):
        pass

    @classmethod
    def contact_detection(cls, env, goal_contact):
        for i in range(env.sim.data.ncon):
            contact = env.sim.data.contact[i]
            achieved_geom1_name = env.sim.model.geom_id2name(contact.geom1)
            achieved_geom2_name = env.sim.model.geom_id2name(contact.geom2)
            if np.all([(achieved_geom1_name in goal_contact), (achieved_geom2_name in goal_contact)]):
                print("contact of " + achieved_geom1_name + " " + achieved_geom2_name)
                return True
            else:
                return False
