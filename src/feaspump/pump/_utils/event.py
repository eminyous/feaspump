from enum import StrEnum


class Event(StrEnum):
    RESET = "reset"
    SETUP = "setup"
    START = "start"
    NO_INTEGERS = "no_integers"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ROUND = "rounding"
    SLACKS = "feasibility"
    PERTURBATION_CHECK = "need_to_perturb"
    LP_CYCLING_CHECK = "lp_cycling_check"
    CYCLING_CHECK = "cycling_check"
    PERTURBED = "perturbed"
    FLIPPED = "flipped"
    ITERATION = "iteration"
    TIME = "time"
