"""
Implementation of SGDRF model.
"""
from typing import Optional

import pyro
import pyro.contrib
import pyro.distributions as dist
import pyro.distributions.util
import pyro.infer
import pyro.nn.module
import pyro.optim
import pyro.util
import torch
import torch.distributions
import torch.distributions.constraints
from pyro.contrib.gp.util import conditional
from pyro.nn.module import PyroParam, pyro_method

from .sgdrf_config import SGDRFConfig
from .subsample import Subsampler

EPSILON = 1e-2
TORCH_DEVICE_CPU = torch.device("cpu")


class SGDRF(pyro.contrib.gp.Parameterized):
    """Streaming Gaussian Dirichlet Random Field model.

    Parameters
    ----------
    config : SGDRFConfig
        SGDRF config object

    Attributes
    ----------
    xu : torch.Tensor
        Sparse inducing points
    dims : int
        The number of spatiotemporal dimensions for this model
    V : int
        The number of observation types
    K : int
        The number of latent Gaussian processes
    M : int
        The number of inducing points
    latent_shape : tuple[int]
        The Pyro shape of the latent Gaussian processes
    max_obs : int
        The maximum number of possible simultaneous categorical observations
    device : torch.device
        Pytorch device (e.g. `torch.device('cuda')`)
    dir_p : torch.Tensor
        The Dirichlet hyperparameters for each entry in the word-topic matrix
    jitter : float
        Small jitter to add to covariance matrix diagonal
    zero_loc : torch.Tensor
        Tensor of all-zeros matching the inducing points
    uloc : pyro.nn.module.PyroParam
        Inducing point mean variational parameter
    uscaletril : pyro.nn.module.PyroParam
        Inducing point lower triangular covariance Cholesky decomposition variational parameter
    word_topic_probs : pyro.nn.module.PyroParam
        Maximum a posteriori word-topic matrix variational parameter
    kernel : pyro.contrib.gp.kernels.Kernel
        Gaussian process kernel
    whiten : bool
        Whether the Gaussian process covariance matrix is whitened
    subsampler : Subsampler
        Subsampling algorithm
    num_particles : int
        Number of parallel posterior latent samples to draw
    objective : pyro.infer.ELBO
        The objective function used during training
    xs : torch.Tensor
        All the locations of past observations
    ws : torch.Tensor
        All the past observations
    optimizer : pyro.optim.PyroOptim
        Stochastic gradient descent optimizer
    svi : pyro.infer.SVI
        Stochastic variational inference helper object
    fail_on_nan_loss : bool
        Whether to raise an exception if training loss is NaN
    n_xs : bool
        The number of past observation locations
    """

    def __init__(self, config: SGDRFConfig):
        """
        Implementation of SGDRF model.
        """
        super().__init__()
        xu_ns = config.xu_ns
        xu_dims = []
        if isinstance(xu_ns, int):
            xu_ns = [xu_ns] * len(config.d_mins)
        for min_d, max_d, xu_n in zip(config.d_mins, config.d_maxs, xu_ns):
            delta_d = max_d - min_d
            dd = delta_d / (xu_n - 1)
            xu_dims.append(torch.arange(start=min_d, end=max_d + dd / 2, step=dd))
        xu = torch.cartesian_prod(*xu_dims).to(config.device).type(torch.float)
        if len(xu.shape) == 1:
            xu = xu.unsqueeze(-1)
        self.xu = xu
        self.dims = len(config.d_maxs)

        self.V = config.V
        self.K = config.K
        self.M = self.xu.size(0)
        self.latent_shape = (self.K,)
        self.max_obs = config.max_obs
        self.device = config.device
        self.dir_p = torch.tensor(
            [[config.dir_p] * config.V] * config.K,
            dtype=torch.float,
            device=config.device,
            requires_grad=True,
        )
        self.jitter = 1e-5
        self.zero_loc = self.xu.new_zeros(self.latent_shape + (self.M,))
        self.uloc = PyroParam(self.zero_loc, dist.constraints.real, event_dim=None)
        self.uscaletril = PyroParam(
            pyro.distributions.util.eye_like(self.xu, self.M).repeat(self.latent_shape + (1, 1)),
            constraint=dist.constraints.stack(
                [dist.constraints.lower_cholesky for _ in range(K)], dim=-3
            ),
            event_dim=None,
        )
        self.word_topic_probs = PyroParam(
            self.dir_p,
            constraint=dist.constraints.stack([dist.constraints.simplex for _ in range(K)], dim=-2),
            event_dim=None,
        )
        self.kernel = config.kernel
        self.whiten = config.whiten
        self.subsampler = config.subsampler
        self.num_particles = config.num_particles
        objective_type = pyro.infer.JitTrace_ELBO if config.jit else pyro.infer.Trace_ELBO
        self.objective = objective_type(
            num_particles=self.num_particles,
            vectorize_particles=True,
            max_plate_nesting=1,
        )
        self.xs = torch.empty(
            0, *self.xu.shape[1:], device=config.device, dtype=torch.float
        )  # type: ignore
        self.ws = torch.empty(0, self.V, device=device, dtype=torch.int)  # type: ignore
        self.optimizer = config.optimizer
        self.svi = pyro.infer.SVI(self.model, self.guide, self.optimizer, self.objective)
        self.fail_on_nan_loss = config.fail_on_nan_loss

        self.n_xs = 0

    def topic_prob(self, xs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Infer or predict topic probabilities.

        Parameters
        ----------
        xs : Optional[torch.Tensor], optional
            The points to predict at, by default self.xs

        Returns
        -------
        torch.Tensor
            Topic probabilities at each point
        """
        f_loc, _ = conditional(
            xs if xs is not None else self.xs,
            self.xu,
            self.kernel,
            self.uloc,
            self.uscaletril,
            full_cov=False,
            whiten=self.whiten,
            jitter=self.jitter,
        )
        topic_probs = torch.softmax(f_loc, -2).squeeze(-2)
        return topic_probs

    def word_topic_prob(self) -> pyro.nn.module.PyroParam:
        """Get the inferred word-topic probability matrix.

        Returns
        -------
        pyro.nn.module.PyroParam
            The inferred word-topic probability matrix (MAP estimate)
        """
        return self.word_topic_probs

    def word_prob(self, xs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Infer or predict word probabilities.

        Parameters
        ----------
        xs : Optional[torch.Tensor], optional
            The points to predict at, by default self.xs

        Returns
        -------
        torch.Tensor
            Word probabilities at each point
        """
        topic_probs = self.topic_prob(xs)
        word_topic_probs = self.word_topic_prob()
        return topic_probs.transpose(-2, -1) @ word_topic_probs

    @staticmethod
    def check_inputs(xs: Optional[torch.Tensor] = None, ws: Optional[torch.Tensor] = None):
        """Check that input location and observation sizes agree in shape.

        Parameters
        ----------
        xs : Optional[torch.Tensor], optional
            The input locations, by default None
        ws : Optional[torch.Tensor], optional
            The input observations, by default None
        """
        if xs is None or ws is None:
            if not ((xs is None) and (ws is None)):
                raise RuntimeError("inputs do not agree")
        else:
            if not (not ((xs is None) and (ws is None)) and (xs.size(0) == ws.size(0))):
                raise RuntimeError("inputs do not agree")

    def process_inputs(self, xs: Optional[torch.Tensor] = None, ws: Optional[torch.Tensor] = None):
        """Ingest new observations.

        Parameters
        ----------
        xs : Optional[torch.Tensor], optional
            New locations, by default None
        ws : Optional[torch.Tensor], optional
            New observations, by default None
        """
        self.check_inputs(xs, ws)
        if xs is not None and ws is not None:
            self.xs = torch.cat([self.xs, xs], dim=0)
            self.ws = torch.cat([self.ws, ws], dim=0)
            self.n_xs += self.xs.shape[0]

    def step(self) -> float:
        """Take a single training step.

        Returns
        -------
        float
            Training loss for this step

        Raises
        ------
        ValueError
            If `self.fail_on_nan_loss` is `True`, raise an error if `loss` is `NaN`
        """
        loss = self.svi.step(self.xs, self.ws, self.subsampler.subsample(self.xs.size(0)))
        if self.fail_on_nan_loss and pyro.util.torch_isnan(loss):
            raise ValueError("loss is NaN")
        return loss  # type: ignore

    @pyro_method
    def model(self, xs: torch.Tensor, ws: torch.Tensor, subsample: torch.Tensor) -> torch.Tensor:
        """Run the stochastic variational inference prior and likelihood model.

        Parameters
        ----------
        xs : torch.Tensor
            Locations of all observations
        ws : torch.Tensor
            Observed categorical data
        subsample : torch.Tensor
            Indices of past observations to use in this training step

        Returns
        -------
        torch.Tensor
            The observations
        """
        self.set_mode("model")
        N = xs.size(0)

        with pyro.plate("topics", self.K, device=xs.device):
            with pyro.util.ignore_jit_warnings():
                Kuu = self.kernel(self.xu).contiguous()
            Luu = torch.linalg.cholesky(Kuu)
            sc = pyro.distributions.util.eye_like(self.xu, self.M) if self.whiten else Luu
            pyro.sample(
                "log_topic_prob_u",
                dist.MultivariateNormal(self.zero_loc, scale_tril=sc),
            )
            word_topic_probs = pyro.sample("word_topic_prob", dist.Dirichlet(self.word_topic_probs))
            with pyro.util.ignore_jit_warnings():
                f_loc, _ = conditional(
                    xs,
                    self.xu,
                    self.kernel,
                    self.uloc,
                    self.uscaletril,
                    Lff=Luu,
                    full_cov=False,
                    whiten=self.whiten,
                    jitter=self.jitter,
                )
        topic_probs = torch.softmax(f_loc, -2).contiguous().squeeze(-2)
        word_probs = topic_probs.transpose(-2, -1) @ word_topic_probs

        with pyro.plate("words", size=N, device=xs.device, subsample=subsample) as i:
            obs = pyro.sample(
                "obs",
                dist.Multinomial(total_count=self.max_obs, probs=word_probs[..., i, :]),
                obs=ws[..., i, :],
            )
        return obs

    @pyro_method
    def guide(self, xs: torch.Tensor, ws: torch.Tensor, subsample: torch.Tensor):
        """Run the stochastic variational inference approximate posterior.

        Parameters
        ----------
        xs : torch.Tensor
            Locations of all observations
        ws : torch.Tensor
            Observed categorical data
        subsample : torch.Tensor
            Indices of past observations to use in this training step
        """
        self.set_mode("guide")
        self._load_pyro_samples()

        with pyro.plate("topics", self.K, device=xs.device):
            pyro.sample(
                "log_topic_prob_u",
                dist.MultivariateNormal(self.uloc, scale_tril=self.uscaletril),
            )
            pyro.sample(
                "word_topic_prob",
                pyro.distributions.Delta(self.word_topic_probs).to_event(1),
            )

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """Produce word probabilities at a given set of locations.

        Parameters
        ----------
        xs : torch.Tensor
            Locations to generate observation probabilities at

        Returns
        -------
        torch.Tensor
            Observation probabilities
        """
        self.set_mode("guide")
        return self.word_prob(xs)
