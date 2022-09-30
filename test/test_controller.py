from typing import Tuple, Union

import numpy as np
import pytest

from fancy_gym.black_box.factory import controller_factory


@pytest.mark.parametrize('ctrl_type', controller_factory.ALL_TYPES)
def test_initialization(ctrl_type: str):
    controller_factory.get_controller(ctrl_type)


@pytest.mark.parametrize('position', [np.zeros(3, ), np.ones(3, ), np.arange(0, 3)])
@pytest.mark.parametrize('velocity', [np.zeros(3, ), np.ones(3, ), np.arange(0, 3)])
def test_velocity(position: np.ndarray, velocity: np.ndarray):
    ctrl = controller_factory.get_controller('velocity')
    a = ctrl(position, velocity, None, None)
    assert np.array_equal(a, velocity)


@pytest.mark.parametrize('position', [np.zeros(3, ), np.ones(3, ), np.arange(0, 3)])
@pytest.mark.parametrize('velocity', [np.zeros(3, ), np.ones(3, ), np.arange(0, 3)])
def test_position(position: np.ndarray, velocity: np.ndarray):
    ctrl = controller_factory.get_controller('position')
    a = ctrl(position, velocity, None, None)
    assert np.array_equal(a, position)


@pytest.mark.parametrize('position', [np.zeros(3, ), np.ones(3, ), np.arange(0, 3)])
@pytest.mark.parametrize('velocity', [np.zeros(3, ), np.ones(3, ), np.arange(0, 3)])
@pytest.mark.parametrize('current_position', [np.zeros(3, ), np.ones(3, ), np.arange(0, 3)])
@pytest.mark.parametrize('current_velocity', [np.zeros(3, ), np.ones(3, ), np.arange(0, 3)])
@pytest.mark.parametrize('p_gains', [0, 1, 0.5, np.zeros(3, ), np.ones(3, ), np.arange(0, 3)])
@pytest.mark.parametrize('d_gains', [0, 1, 0.5, np.zeros(3, ), np.ones(3, ), np.arange(0, 3)])
def test_pd(position: np.ndarray, velocity: np.ndarray, current_position: np.ndarray, current_velocity: np.ndarray,
            p_gains: Union[float, Tuple], d_gains: Union[float, Tuple]):
    ctrl = controller_factory.get_controller('motor', p_gains=p_gains, d_gains=d_gains)
    assert np.array_equal(ctrl.p_gains, p_gains)
    assert np.array_equal(ctrl.d_gains, d_gains)

    a = ctrl(position, velocity, current_position, current_velocity)
    pd = p_gains * (position - current_position) + d_gains * (velocity - current_velocity)
    assert np.array_equal(a, pd)


@pytest.mark.parametrize('pos_vel', [(np.ones(3, ), np.ones(4, )),
                                     (np.ones(4, ), np.ones(3, )),
                                     (np.ones(4, ), np.ones(4, ))])
def test_pd_invalid_shapes(pos_vel: Tuple[np.ndarray, np.ndarray]):
    position, velocity = pos_vel
    ctrl = controller_factory.get_controller('motor')
    with pytest.raises(ValueError):
        ctrl(position, velocity, np.ones(3, ), np.ones(3, ))


@pytest.mark.parametrize('position', [np.zeros(3, ), np.ones(3, ), np.arange(0, 3)])
@pytest.mark.parametrize('current_position', [np.zeros(3, ), np.ones(3, ), np.arange(0, 3)])
@pytest.mark.parametrize('gripper_pos', [0, 1, 0.5])
def test_metaworld(position: np.ndarray, current_position: np.ndarray, gripper_pos: float):
    ctrl = controller_factory.get_controller('metaworld')

    position_grip = np.append(position, gripper_pos)
    c_position_grip = np.append(current_position, -1)
    a = ctrl(position_grip, None, c_position_grip, None)
    assert a[-1] == gripper_pos
    assert np.array_equal(a[:-1], position - current_position)


def test_metaworld_invalid_shapes():
    ctrl = controller_factory.get_controller('metaworld')
    with pytest.raises(ValueError):
        ctrl(np.ones(4, ), None, np.ones(3, ), None)
