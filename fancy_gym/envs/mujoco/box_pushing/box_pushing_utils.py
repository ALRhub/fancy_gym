import numpy as np
from matplotlib import pyplot as plt


# joint constraints for Franka robot
q_max = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])

q_dot_max = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])
q_torque_max = np.array([90., 90., 90., 90., 12., 12., 12.])
#
desired_rod_quat = np.array([0.0, 1.0, 0.0, 0.0])

def skew(x):
    """
    Returns the skew-symmetric matrix of x
    param x: 3x1 vector
    """
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])

def get_quaternion_error(curr_quat, des_quat):
    """
    Calculates the difference between the current quaternion and the desired quaternion.
    See Siciliano textbook page 140 Eq 3.91

    param curr_quat: current quaternion
    param des_quat: desired quaternion
    return: difference between current quaternion and desired quaternion
    """
    return curr_quat[0] * des_quat[1:] - des_quat[0] * curr_quat[1:] - skew(des_quat[1:]) @ curr_quat[1:]

def rotation_distance(p: np.array, q: np.array):
    """
    Calculates the rotation angular between two quaternions
    param p: quaternion
    param q: quaternion
    theta: rotation angle between p and q (rad)
    """
    assert p.shape == q.shape, "p and q should be quaternion"
    theta = 2 * np.arccos(abs(p @ q))
    return theta


def rot_to_quat(theta, axis):
    """
    Converts rotation angle along an axis to quaternion
    param theta: rotation angle (rad)
    param axis: rotation axis
    return: quaternion
    """
    quant = np.zeros(4)
    quant[0] = np.sin(theta / 2.)
    quant[1:] = np.cos(theta / 2.) * axis
    return quant


def img_to_world_testing(env, pixel_pos=None, cam="rgbd"):
    """
    Render steps and test image to world function. The image position and the real world
    position are rendered separately. Image position is drawn on the image while the real
    world point is rendered in the simulation.

    Args:
        env (MujocoEnv): environment, function implemented for BoxPushingBin specifically
        pixel_pos (np.array): array with two elements, if None sample position randomly
        cam (str): camera name to test
    """
    def draw_cross_at_pos(array, pos, size: int = 3):
        fill = np.zeros(3) if array.shape[-1] == 3 else 0.98
        array[pos[0], pos[1]] = fill
        for i in range(1, size):
            high = np.stack([array.shape[0] -1 , array.shape[1] - 1] * 4).reshape(4, 2)
            pos_ = np.stack([pos] * 4) + np.array([[1, -1], [1, 1], [-1, 1], [-1, -1]])
            pos_ = np.clip(pos_, 0, high)
            for p in pos_: array[tuple(p)] = fill

        return array

    depth = env.render(
        mode="depth_array", width=env.width, height=env.height, camera_name=cam
    ).copy()

    if pixel_pos is not None:
        assert pixel_pos[0] < env.width and pixel_pos[1] < env.height and\
               pixel_pos[0] >= 0 and pixel_pos[1] >= 0
    else:
        pixel_pos = np.random.randint(np.zeros(2), np.array([env.height, env.width]))

    # Convert img pixel to simulation coordinate, draw that coordinate in simulation
    depth_at_pos = depth[tuple(pixel_pos)]
    world_coords = env.img_to_world(np.append(pixel_pos, depth_at_pos))
    env.data.site("ref").xpos = world_coords
    print("World coordinates (x, y, z): ", world_coords)

    rgb = env.render(
        mode="rgb_array", width=env.width, height=env.height, camera_name=cam
    ).copy()
    depth = env.render(
        mode="depth_array", width=env.width, height=env.height, camera_name=cam
    ).copy()

    env.data.site("ref").xpos = np.array([0, 0, -2]) # hide again

    # draw pixels on the observation
    rgb = draw_cross_at_pos(rgb, pixel_pos)
    depth = draw_cross_at_pos(depth, pixel_pos)

    fig = plt.figure(figsize=(12, 6))
    fig.add_subplot(1, 2, 1)
    plt.imshow(rgb)
    fig.add_subplot(1, 2, 2)
    plt.imshow(depth)
    plt.show()


def set_tcp_testing(step, env, render=False, thresh=0.2):
    """
    Tests hard setting the tcp position by sampling a point inside the boundaries and
    call set_tcp_pos that uses IK to find the correct joint position to get to the desired
    position. The function also renders a reference point in the desired position.

    Args:
        step (0 or 1): 0 for first step (sample & set robot position), 1 for reset robot
        env (BoxPushingBin): environment
        render (bool): render steps
    """
    if step == 0:
        pos_bounds = np.array([[0.4, -0.3, 0.01], [0.8, 0.3, 0.15]])
        world_point = env.np_random.uniform(low=pos_bounds[0], high=pos_bounds[1])
        env.data.site("ref").xpos = world_point
        env.render(mode="human") if render else None
        env.set_tcp_pos(world_point, hard_set=True)
        env.data.site("ref").xpos = world_point
        env.render(mode="human") if render else None
        return 1 if np.abs(env.data.body("tcp").xpos - world_point).sum() > thresh else 0
    if step == 1:
        env.reset_robot_pos()
