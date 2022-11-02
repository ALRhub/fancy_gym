import os

import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import MujocoEnv

import mujoco

MAX_EPISODE_STEPS_TABLE_TENNIS = 250

CONTEXT_BOUNDS_2DIMS = np.array([[-1.2, -0.6], [-0.2, 0.0]])
CONTEXT_BOUNDS_4DIMS = np.array([[-1.2, -0.6, -1.0, -0.65],
                                 [-0.2, 0.6, -0.2, 0.65]])


class TableTennisEnv(MujocoEnv, utils.EzPickle):
    """
    7 DoF table tennis environment
    """

    def __init__(self, ctxt_dim: int = 2, frame_skip: int = 4):
        utils.EzPickle.__init__(**locals())
        self._steps = 0

        self.hit_ball = False
        self.ball_land_on_table = False
        self._id_set = False
        self.ball_landing_pos = None
        MujocoEnv.__init__(self,
                           model_path=os.path.join(os.path.dirname(__file__), "assets", "xml", "table_tennis_env.xml"),
                           frame_skip=frame_skip,
                           mujoco_bindings="mujoco")
        if ctxt_dim == 2:
            self.context_bounds = CONTEXT_BOUNDS_2DIMS
        elif ctxt_dim == 4:
            self.context_bounds = CONTEXT_BOUNDS_4DIMS
        else:
            raise NotImplementedError

        self.action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)

    def _set_ids(self):
        self._floor_contact_id = self.model.geom("floor").bodyid[0]
        self._ball_contact_id = self.model.geom("target_ball_contact").bodyid[0]
        self._bat_front_id = self.model.geom("bat").bodyid[0]
        self._bat_back_id = self.model.geom("bat_back").bodyid[0]
        self._table_contact_id = self.model.geom("table_tennis_table").bodyid[0]
        self._id_set = True

    def step(self, action):
        if not self._id_set:
            self._set_ids()

        unstable_simulation = False

        done = False

        for _ in range(self.frame_skip):
            try:
                self.do_simulation(action, self.frame_skip)
            except Exception as e:
                print("Simulation get unstable return with MujocoException: ", e)
                unstable_simulation = True

            if not self.hit_ball:
                self.hit_ball = self._contact_checker(self._ball_contact_id, self._bat_front_id) or \
                                self._contact_checker(self._ball_contact_id, self._bat_back_id)
                if not self.hit_ball:
                    ball_land_on_floor_no_hit = self._contact_checker(self._ball_contact_id, self._floor_contact_id)
                    if ball_land_on_floor_no_hit:
                        self.ball_landing_pos = self.data.body("target_ball").xpos.copy()
                        done = True
            if self.hit_ball and not self.ball_contact_after_hit:
                if not self.ball_contact_after_hit:
                    if self._contact_checker(self._ball_contact_id, self._floor_contact_id):  # first check contact with floor
                        self.ball_contact_after_hit = True
                        self.ball_landing_pos = self.sim.data.geom("target_ball_contact").xpos.copy()
                    elif self._contact_checker(self._ball_contact_id, self._table_contact_id):  # second check contact with table
                        self.ball_contact_after_hit = True
                        self.ball_landing_pos = self.sim.data.geom("target_ball_contact").xpos.copy()
                        if self.ball_landing_pos[0] < 0.:  # ball lands on the opponent side
                            self.ball_return_success = True

        self._steps += 1
        episode_end = True if self._steps >= MAX_EPISODE_STEPS_TABLE_TENNIS else False

        obs = self._get_obs()

        return obs, 0., False, {}

    def _contact_checker(self, id_1, id_2):
        for coni in range(0, self.data.ncon):
            con = self.data.contact[coni]
            if (con.geom1 == id_1 and con.geom2 == id_2) or (con.geom1 == id_2 and con.geom2 == id_1):
                return True
        return False

    def reset_model(self):
        self._steps = 0
        new_context = self._sample_context()
        self.data.joint("tar_x").qpos = new_context[0]
        self.data.joint("tar_y").qpos = new_context[1]
        self.data.joint("tar_z").qvel = 2.

        self.ball_landing_pos = None
        self.hit_ball = False
        return self._get_obs()

    def _sample_context(self):
        return self.np_random.uniform(low=self.context_bounds[0],
                                      high=self.context_bounds[1])

    def _get_obs(self):
        obs = np.concatenate([
            self.data.qpos.flat[:7],
            self.data.qvel.flat[:7],
        ])
        return obs


if __name__ == "__main__":
    env = TableTennisEnv()
    env.reset()
    for _ in range(1000):
        for _ in range(200):
            env.render("human")
            env.step(env.action_space.sample())
