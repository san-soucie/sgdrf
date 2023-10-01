"""
Wrapper for pyro optimizers.
"""
from enum import Enum, auto

import pyro.optim as optim
from pyro.optim import PyroOptim


class OptimizerType(Enum):
    """A wrapper for ::py:class:`pyro.optim.PyroOptim` optimizers."""

    Adadelta = auto()
    """::py:class:`pyro.optim.Adadelta`"""
    Adagrad = auto()
    """::py:class:`pyro.optim.Adagrad`"""
    Adam = auto()
    """::py:class:`pyro.optim.Adam`"""
    AdamW = auto()
    """::py:class:`pyro.optim.AdamW`"""
    Adamax = auto()
    """::py:class:`pyro.optim.Adamax`"""
    SGD = auto()
    """::py:class:`pyro.optim.SGD`"""

    def instantiate(self, lr: float = 0.001, clip_norm: float = 10.0) -> PyroOptim:
        """Instantiate the optimizer.

        Parameters
        ----------
        lr : float, optional
            Optimizer learning rate, by default 0.001
        clip_norm : float, optional
            Optimizer gradient norm maximum, by default 10.0

        Returns
        -------
        PyroOptim
            The instantiated optimizer
        """
        return getattr(optim, self._name_)({"lr": lr}, clip_args={"clip_norm": clip_norm})
