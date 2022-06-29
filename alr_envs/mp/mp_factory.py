from mp_pytorch.mp.dmp import DMP
from mp_pytorch.mp.promp import ProMP
from mp_pytorch.mp.idmp import IDMP

from mp_pytorch.basis_gn.basis_generator import BasisGenerator

ALL_TYPES = ["promp", "dmp", "idmp"]


def get_trajectory_generator(
        trajectory_generator_type: str, action_dim: int, basis_generator: BasisGenerator, **kwargs
        ):
    trajectory_generator_type = trajectory_generator_type.lower()
    if trajectory_generator_type == "promp":
        return ProMP(basis_generator, action_dim, **kwargs)
    elif trajectory_generator_type == "dmp":
        return DMP(basis_generator, action_dim, **kwargs)
    elif trajectory_generator_type == 'idmp':
        return IDMP(basis_generator, action_dim, **kwargs)
    else:
        raise ValueError(f"Specified movement primitive type {trajectory_generator_type} not supported, "
                         f"please choose one of {ALL_TYPES}.")