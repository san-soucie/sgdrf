from enum import Flag, auto


class SubsampleType(Flag):
    """A simple wrapper for the various subsampling strategies."""

    latest = auto()
    """Always sample the most recent `N` observations"""
    uniform = auto()
    """Sample uniformly at random from all past observations."""
    exponential = auto()
    """Sample with probability proportional to
    :math:`exp[{\\alpha}(t-{\\tau})]`"""
    exponential_plus_uniform = exponential | uniform
    """Sample with probabilities representing a weighted sum of exponential &
    uniform strategies."""
    exponential_plus_latest = exponential | latest
    """Sample with probabilities representing a weighted sum of exponential &
    latest strategies."""
    uniform_plus_latest = uniform | latest
    """Sample with probabilities representing a weighted sum of uniform &
    latest strategies."""
