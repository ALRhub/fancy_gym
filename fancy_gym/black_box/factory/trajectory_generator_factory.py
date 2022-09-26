from mp_pytorch.basis_gn import BasisGenerator
from mp_pytorch.mp import ProDMP, DMP, ProMP

ALL_TYPES = ["promp", "dmp", "idmp"]


def get_trajectory_generator(
        trajectory_generator_type: str, action_dim: int, basis_generator: BasisGenerator, **kwargs
):
    trajectory_generator_type = trajectory_generator_type.lower()
    if trajectory_generator_type == "promp":
        return ProMP(basis_generator, action_dim, **kwargs)
    elif trajectory_generator_type == "dmp":
        return DMP(basis_generator, action_dim, **kwargs)
    elif trajectory_generator_type == 'prodmp':
        from mp_pytorch.basis_gn import ProDMPBasisGenerator
        assert isinstance(basis_generator, ProDMPBasisGenerator)
        return ProDMP(basis_generator, action_dim, **kwargs)
    else:
        raise ValueError(f"Specified movement primitive type {trajectory_generator_type} not supported, "
                         f"please choose one of {ALL_TYPES}.")
