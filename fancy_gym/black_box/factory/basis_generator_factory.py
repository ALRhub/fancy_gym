from mp_pytorch import PhaseGenerator, NormalizedRBFBasisGenerator, ZeroStartNormalizedRBFBasisGenerator
from mp_pytorch.basis_gn.rhytmic_basis import RhythmicBasisGenerator

ALL_TYPES = ["rbf", "zero_rbf", "rhythmic"]


def get_basis_generator(basis_generator_type: str, phase_generator: PhaseGenerator, **kwargs):
    basis_generator_type = basis_generator_type.lower()
    if basis_generator_type == "rbf":
        return NormalizedRBFBasisGenerator(phase_generator, **kwargs)
    elif basis_generator_type == "zero_rbf":
        return ZeroStartNormalizedRBFBasisGenerator(phase_generator, **kwargs)
    elif basis_generator_type == "rhythmic":
        return RhythmicBasisGenerator(phase_generator, **kwargs)
    else:
        raise ValueError(f"Specified basis generator type {basis_generator_type} not supported, "
                         f"please choose one of {ALL_TYPES}.")
