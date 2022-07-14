from mp_pytorch.phase_gn import LinearPhaseGenerator, ExpDecayPhaseGenerator

# from mp_pytorch.phase_gn.rhythmic_phase_generator import RhythmicPhaseGenerator
# from mp_pytorch.phase_gn.smooth_phase_generator import SmoothPhaseGenerator

ALL_TYPES = ["linear", "exp", "rhythmic", "smooth"]


def get_phase_generator(phase_generator_type, **kwargs):
    phase_generator_type = phase_generator_type.lower()
    if phase_generator_type == "linear":
        return LinearPhaseGenerator(**kwargs)
    elif phase_generator_type == "exp":
        return ExpDecayPhaseGenerator(**kwargs)
    elif phase_generator_type == "rhythmic":
        raise NotImplementedError()
        # return RhythmicPhaseGenerator(**kwargs)
    elif phase_generator_type == "smooth":
        raise NotImplementedError()
        # return SmoothPhaseGenerator(**kwargs)
    else:
        raise ValueError(f"Specified phase generator type {phase_generator_type} not supported, "
                         f"please choose one of {ALL_TYPES}.")
