"""
Classes for subsampling strategies.
"""
from abc import ABCMeta, abstractmethod
from typing import List, Optional, Union

import numpy as np
import pyro.distributions as dist
import torch
from torch.nn.functional import normalize


class Subsampler(metaclass=ABCMeta):
    """
    An abstract class for SGDRF subsampling strategies.
    """

    def __init__(self, n: int, device: torch.device):
        self.n = n
        self.device = device

    def rtp1(self, t: int) -> torch.Tensor:
        return torch.arange(t + 1, device=self.device).type(torch.float)

    @abstractmethod
    def dist(self, t: int) -> torch.Tensor:
        raise NotImplementedError

    def subsample(self, t: int) -> torch.Tensor:
        return dist.Categorical(self.dist(t)).sample(torch.Size([self.n]))


class MixingSubsampler(Subsampler):
    def __init__(self, n: int, device: torch.device, weight: float, s1: Subsampler, s2: Subsampler):
        super().__init__(n=n, device=device)
        self.weight = weight
        self.s1 = s1
        self.s2 = s2

    def dist(self, t: int):
        return normalize(self.s1.dist(t) + self.s2.dist(t), p=1.0, dim=0)


class ExponentialSubsampler(Subsampler):
    def __init__(self, n: int, device: torch.device, exp: float):
        super().__init__(n=n, device=device)
        self.exp = torch.tensor(exp, device=device, dtype=torch.float)

    def dist(self, t: int) -> torch.Tensor:
        exp_vals = self.exp * (
            torch.arange(t + 1, device=self.device).type(torch.float)
            - torch.tensor(t, device=self.device, dtype=torch.float)
        )
        return normalize(torch.exp(exp_vals), p=1.0, dim=0)


class UniformSubsampler(Subsampler):
    def dist(self, t: int) -> torch.Tensor:
        return normalize(torch.ones((t + 1,), device=self.device, dtype=torch.float), p=1.0, dim=0)


class LatestSubsampler(Subsampler):
    def dist(self, t: int) -> torch.Tensor:
        arr = torch.zeros((t + 1,), device=self.device, dtype=torch.float)
        arr[-1] += 1.0
        return normalize(arr, p=1.0, dim=0)
