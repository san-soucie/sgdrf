"""
Classes for subsampling strategies.
"""
from abc import ABCMeta, abstractmethod

import pyro.distributions as dist
import torch
from torch.nn.functional import normalize


class Subsampler(metaclass=ABCMeta):
    """
    An abstract class for SGDRF subsampling strategies.

    Parameters
    ----------
    n : int
        Number of subsamples to draw
    device : torch.device
        Pytorch device to use

    Attributes
    ----------
    n : int
        Number of subsamples to draw
    device : torch.device
        Pytorch device to use
    """

    def __init__(self, n: int, device: torch.device):
        self.n = n
        self.device = device

    @abstractmethod
    def dist(self, t: int) -> torch.Tensor:
        """
        Get distribution for this subsampler, for `t` observations.

        Parameters
        ----------
        t : int
            The number of observations to get a distribution for

        Returns
        -------
        torch.Tensor
            Tensor of topic probabilities of length `t`
        """
        raise NotImplementedError

    def subsample(self, t: int) -> torch.Tensor:
        """
        Generate a subsample, for `t` observations.

        Parameters
        ----------
        t : int
            The number of observations to get a subsample for

        Returns
        -------
        torch.Tensor
            Tensor of subsample
        """
        return dist.Categorical(self.dist(t)).sample(torch.Size([self.n]))


class MixingSubsampler(Subsampler):
    """
    A subsampling strategy that mixes two other subsampling strategies with a given weight.

    Parameters
    ----------
    n : int
        Number of subsamples to draw
    device : torch.device
        Pytorch device to use
    weight : float
        Weight associated with subsampler `s1` (`s2` is associated with `1 - weight`)
    s1 : Subsampler
        The first subsampler to mix
    s2: Subsampler
        The second subsampler to mix

    Attributes
    ----------
    n : int
        Number of subsamples to draw
    device : torch.device
        Pytorch device to use
    weight : float
        Weight associated with subsampler `s1` (`s2` is associated with `1 - weight`)
    s1 : Subsampler
        The first subsampler to mix
    s2: Subsampler
        The second subsampler to mix
    """

    def __init__(self, n: int, device: torch.device, weight: float, s1: Subsampler, s2: Subsampler):
        super().__init__(n=n, device=device)
        self.weight = weight
        self.s1 = s1
        self.s2 = s2

    def dist(self, t: int):
        """
        Get distribution for this subsampler, for `t` observations.

        Parameters
        ----------
        t : int
            The number of observations to get a distribution for

        Returns
        -------
        torch.Tensor
            Tensor of topic probabilities of length `t`
        """
        arr = torch.tensor(
            self.weight * self.s1.dist(t) + (1 - self.weight) * self.s2.dist(t),
            device=self.device,
            dtype=torch.float,
        )
        return normalize(arr, p=1.0, dim=0)


class ExponentialSubsampler(Subsampler):
    """
    A subsampling strategy that has an exponential decay, with larger probabilities being more recent.

    Parameters
    ----------
    n : int
        Number of subsamples to draw
    device : torch.device
        Pytorch device to use
    exp : float
        Exponential parameter

    Attributes
    ----------
    n : int
        Number of subsamples to draw
    device : torch.device
        Pytorch device to use
    exp : float
        Exponential parameter
    """

    def __init__(self, n: int, device: torch.device, exp: float):
        super().__init__(n=n, device=device)
        self.exp = torch.tensor(exp, device=device, dtype=torch.float)

    def dist(self, t: int) -> torch.Tensor:
        """
        Get distribution for this subsampler, for `t` observations.

        Parameters
        ----------
        t : int
            The number of observations to get a distribution for

        Returns
        -------
        torch.Tensor
            Tensor of topic probabilities of length `t`
        """
        exp_vals = self.exp * (
            torch.arange(t + 1, device=self.device).type(torch.float)
            - torch.tensor(t, device=self.device, dtype=torch.float)
        )
        return normalize(torch.exp(exp_vals), p=1.0, dim=0)


class UniformSubsampler(Subsampler):
    """
    A subsampling strategy that has uniform values for all observations.

    Parameters
    ----------
    n : int
        Number of subsamples to draw
    device : torch.device
        Pytorch device to use

    Attributes
    ----------
    n : int
        Number of subsamples to draw
    device : torch.device
        Pytorch device to use
    """

    def dist(self, t: int) -> torch.Tensor:
        """
        Get distribution for this subsampler, for `t` observations.

        Parameters
        ----------
        t : int
            The number of observations to get a distribution for

        Returns
        -------
        torch.Tensor
            Tensor of topic probabilities of length `t`
        """
        return normalize(torch.ones((t + 1,), device=self.device, dtype=torch.float), p=1.0, dim=0)


class LatestSubsampler(Subsampler):
    """
    A subsampling strategy that samples the `n` most recent observations with probability `1`.

    Parameters
    ----------
    n : int
        Number of subsamples to draw
    device : torch.device
        Pytorch device to use

    Attributes
    ----------
    n : int
        Number of subsamples to draw
    device : torch.device
        Pytorch device to use
    """

    def dist(self, t: int) -> torch.Tensor:
        """
        Get distribution for this subsampler, for `t` observations.

        Parameters
        ----------
        t : int
            The number of observations to get a distribution for

        Returns
        -------
        torch.Tensor
            Tensor of topic probabilities of length `t`
        """
        arr = torch.zeros((t + 1,), device=self.device, dtype=torch.float)
        arr[-1] += 1.0
        return normalize(arr, p=1.0, dim=0)
