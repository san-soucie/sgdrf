from dataclasses import dataclass
from typing import Collection, Union

import pyro.contrib.gp.kernels
import pyro.optim
import torch

from .subsample import Subsampler


@dataclass
class SGDRFConfig:
    """Streaming Gaussian Dirichlet Random Field configuration holder.

    Parameters
    ----------
    xu_ns : Union[int, Collection[int]]
        Number of inducing points, either one for each spatiotemporal dimension or a single number.
    d_mins : Collection[float]
        A collection containing the minimum extent of each dimension
    d_maxs : Collection[float]
        A collection containing the maximum extent of each dimension
    V : int
        The number of observation types
    K : int
        The number of latent Gaussian processes
    max_obs : int
        The maximum number of possible simultaneous categorical observations
    dir_p : float
        Initial uniform Dirichlet hyperparameter
    kernel : pyro.contrib.gp.kernels.Kernel
        Latent Gaussian process kernel
    optimizer : pyro.optim.PyroOptim
        Stochastic gradient descent optimization algorithm
    subsampler : Subsampler
        Subsampling algorithm
    device : torch.device, optional
        Pytorch device (e.g. `torch.device('cuda')`), by default torch.device("cpu")
    whiten : bool, optional
        Whether the Gaussian process covariance matrix is whitened, by default False
    fail_on_nan_loss : bool, optional
        Whether to raise an exception if training loss is NaN, by default True
    num_particles : int, optional
        Number of parallel posterior latent samples to draw, by default 1
    jit : bool, optional
        Whether to JIT-compile the model and guide, by default False
    """

    xu_ns: Union[int, Collection[int]]
    d_mins: Collection[float]
    d_maxs: Collection[float]
    V: int
    K: int
    max_obs: int
    dir_p: float
    kernel: pyro.contrib.gp.kernels.Kernel
    optimizer: pyro.optim.PyroOptim
    subsampler: Subsampler
    device: torch.device = torch.device("cpu")
    whiten: bool = False
    fail_on_nan_loss: bool = True
    num_particles: int = 1
    jit: bool = False
