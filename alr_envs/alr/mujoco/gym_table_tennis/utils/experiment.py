import numpy as np
from gym.utils import seeding
from alr_envs.alr.mujoco.gym_table_tennis.utils.util import read_yaml, read_json
from pathlib import Path


def ball_initialize(random=False, scale=False, context_range=None, scale_value=None):
    if random:
        if scale:
            # if scale_value is None:
            scale_value = context_scale_initialize(context_range)
            v_x, v_y, v_z = [2.5, 2, 0.5] * scale_value
            dx = 1
            dy = 0
            dz = 0.05
        else:
            seed = None
            np_random, seed = seeding.np_random(seed)
            dx = np_random.uniform(-0.1, 0.1)
            dy = np_random.uniform(-0.1, 0.1)
            dz = np_random.uniform(-0.1, 0.1)

            v_x = np_random.uniform(1.7, 1.8)
            v_y = np_random.uniform(0.7, 0.8)
            v_z = np_random.uniform(0.1, 0.2)
        # print(dx, dy, dz, v_x, v_y, v_z)
    # else:
    #     dx = -0.1
    #     dy = 0.05
    #     dz = 0.05
    #     v_x = 1.5
    #     v_y = 0.7
    #     v_z = 0.06
    # initial_x = -0.6 + dx
    # initial_y = -0.3 + dy
    # initial_z = 0.8 + dz
    else:
        if scale:
            v_x, v_y, v_z = [2.5, 2, 0.5] * scale_value
        else:
            v_x = 2.5
            v_y = 2
            v_z = 0.5
        dx = 1
        dy = 0
        dz = 0.05

    initial_x = 0 + dx
    initial_y = -0.2 + dy
    initial_z = 0.3 + dz
    # print("initial ball state: ", initial_x, initial_y, initial_z, v_x, v_y, v_z)
    initial_ball_state = np.array([initial_x, initial_y, initial_z, v_x, v_y, v_z])
    return initial_ball_state


def context_scale_initialize(range):
    """

    Returns:

    """
    low, high = range
    scale = np.random.uniform(low, high, 1)
    return scale


def config_handle_generation(config_file_path):
    """Generate config handle for multiprocessing

    Args:
        config_file_path:

    Returns:

    """
    cfg_fname = Path(config_file_path)
    # .json and .yml file
    if cfg_fname.suffix == ".json":
        config = read_json(cfg_fname)
    elif cfg_fname.suffix == ".yml":
        config = read_yaml(cfg_fname)

    return config
