"""
Implementation of SGDRF model.
"""
from typing import Collection, Dict, Optional, Union

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
from torch.nn.functional import normalize

from .kernel import KernelType
from .optimizer import OptimizerType
from .subsample import SubsampleType

EPSILON = 1e-2
TORCH_DEVICE_CPU = torch.device("cpu")


class SGDRF(pyro.contrib.gp.Parameterized):
    """Streaming Gaussian Dirichlet Random Field model.

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
    kernel_type : KernelType, optional
        Latent Gaussian process kernel type, by default KernelType.Matern52
    kernel_lengthscale : float, optional
        Latent Gaussian process kernel lengthscale, by default 1.0
    kernel_variance : float, optional
        Latent Gaussian process kernel variance, by default 1.0
    optimizer_type : OptimizerType, optional
        Stochastic gradient descent optimization algorithm, by default OptimizerType.Adam
    optimizer_lr : float, optional
        Stochastic gradient descent optimization learning rate, by default 0.01
    optimizer_clip_norm : float, optional
        Stochastic gradient descent optimization maximum gradient norm, by default 10.0
    device : torch.device, optional
        Pytorch device (e.g. `torch.device('cuda')`), by default torch.device("cpu")
    subsample_n : int, optional
        Number of past observations to subsample in a single training step, by default 1
    subsample_type : SubsampleType, optional
        Subsampling strategy to use, by default SubsampleType.uniform
    subsample_params : Optional[dict[str, float]], optional
        A dictionary containing the subsample parameters `exponential` and `weight`,
        by default `{"exponential": 0.1, "weight": 0.5}`
    whiten : bool, optional
        Whether the Gaussian process covariance matrix is whitened, by default False
    fail_on_nan_loss : bool, optional
        Whether or not to raise an exception if training loss is NaN, by default True
    num_particles : int, optional
        Number of parallel posterior latent samples to draw, by default 1
    jit : bool, optional
        Whether or not to JIT-compile the model and guide, by default False

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
    subsample_type : SubsampleType
        Subsampling strategy to use
    subsample_params : dict[str, float]
        A dictionary containing the subsample parameters `exponential` and `weight`
    subsample_n : int
        Number of past observations to subsample in a single training step
    subsample_n_tensor : torch.Tensor
        Number of past observations to subsample in a single training step, as a tensor
    num_particles : int
        Number of parallel posterior latent samples to draw
    objective : pyro.infer.ELBO
        The objective function used during training
    xs : torch.Tensor
        All of the locations of past observations
    ws : torch.Tensor
        All of the past observations
    optimizer : pyro.optim.PyroOptim
        Stochastic gradient descent optimizer
    svi : pyro.infer.SVI
        Stochastic variational inference helper object
    fail_on_nan_loss : bool
        Whether or not to raise an exception if training loss is NaN
    n_xs : bool
        The number of past observation locations
    """

    def __init__(
        self,
        xu_ns: Union[int, Collection[int]],
        d_mins: Collection[float],
        d_maxs: Collection[float],
        V: int,
        K: int,
        max_obs: int,
        dir_p: float,
        kernel_type: KernelType = KernelType.Matern52,
        kernel_lengthscale: float = 1.0,
        kernel_variance: float = 1.0,
        optimizer_type: OptimizerType = OptimizerType.Adam,
        optimizer_lr: float = 0.01,
        optimizer_clip_norm: float = 10.0,
        device: torch.device = TORCH_DEVICE_CPU,
        subsample_n: int = 1,
        subsample_type: SubsampleType = SubsampleType.uniform,
        subsample_params: Optional[Dict[str, float]] = None,
        whiten: bool = False,
        fail_on_nan_loss: bool = True,
        num_particles: int = 1,
        jit: bool = False,
    ):
        """
        Implementation of SGDRF model.
        """
        super().__init__()

        xu_dims = []
        if isinstance(xu_ns, int):
            xu_ns = [xu_ns] * len(d_mins)
        for min_d, max_d, xu_n in zip(d_mins, d_maxs, xu_ns):
            delta_d = max_d - min_d
            dd = delta_d / (xu_n - 1)
            xu_dims.append(torch.arange(start=min_d, end=max_d + dd / 2, step=dd))
        xu = torch.cartesian_prod(*xu_dims).to(device).type(torch.float)
        if len(xu.shape) == 1:
            xu = xu.unsqueeze(-1)
        self.xu = xu
        self.dims = len(d_maxs)

        self.V = V
        self.K = K
        self.M = self.xu.size(0)
        self.latent_shape = (self.K,)
        self.max_obs = max_obs
        self.device = device
        self.dir_p = torch.tensor(
            [[dir_p] * V] * K, dtype=torch.float, device=device, requires_grad=True
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
        self.kernel = kernel_type.instantiate(
            input_dim=self.xu.size(1),
            variance=torch.tensor(kernel_variance, dtype=torch.float, device=device),
            lengthscale=torch.tensor(kernel_lengthscale, dtype=torch.float, device=device),
        )

        self.whiten = whiten
        if subsample_params is None:
            subsample_params = {"exponential": 0.1, "weight": 0.5}
        self.subsample_type = subsample_type
        self.subsample_params = subsample_params
        self.subsample_n = subsample_n
        self.subsample_n_tensor = torch.tensor(
            self.subsample_n, dtype=torch.int, device=self.device
        )
        self.num_particles = num_particles
        objective_type = pyro.infer.JitTrace_ELBO if jit else pyro.infer.Trace_ELBO
        self.objective = objective_type(
            num_particles=self.num_particles,
            vectorize_particles=True,
            max_plate_nesting=1,
        )
        self.xs = torch.empty(
            0, *self.xu.shape[1:], device=device, dtype=torch.float
        )  # type: ignore
        self.ws = torch.empty(0, self.V, device=device, dtype=torch.int)  # type: ignore
        self.optimizer = optimizer_type.instantiate(lr=optimizer_lr, clip_norm=optimizer_clip_norm)
        self.svi = pyro.infer.SVI(self.model, self.guide, self.optimizer, self.objective)
        self.fail_on_nan_loss = fail_on_nan_loss

        self.n_xs = 0

    def subsample(self, t: Optional[int] = None) -> torch.Tensor:
        """Generate a subsampling index tensor.

        Parameters
        ----------
        t : Optional[int], optional
            The index to use as `most recent` for sampling, by default self.xs.size(0) - 1

        Returns
        -------
        torch.Tensor
            A tensor representing the indices of the past observations chosen for this subsample

        Raises
        ------
        ValueError
            Raises a ValueError if the subsample type is invalid.
        """
        n = self.subsample_n
        t = t if t is not None else self.xs.size(0) - 1
        if (self.subsample_type == "full") or (t <= n):
            return torch.tensor(list(range(t + 1)), dtype=torch.long)

        rtp1 = torch.arange(t + 1, device=self.device).type(torch.float)
        latest = normalize(
            torch.tensor(
                data=[0 for _ in range(t)]
                + [
                    1,
                ],
                device=self.device,
                dtype=torch.float,
            ),
            p=1.0,
            dim=0,
        )
        exponential = normalize(
            torch.exp(-(t - rtp1) * self.subsample_params["exponential"]), p=1.0, dim=0
        )
        uniform = normalize(0.0 * rtp1 + 1.0, p=1.0, dim=0)
        if t < 1 or self.subsample_type == SubsampleType.latest:
            probs = latest
        elif self.subsample_type == SubsampleType.exponential:
            probs = exponential
        elif self.subsample_type == SubsampleType.uniform:
            probs = uniform
        elif self.subsample_type == SubsampleType.exponential_plus_uniform:
            probs = normalize(
                exponential * self.subsample_params["weight"]
                + uniform * (1.0 - self.subsample_params["weight"]),
                p=1.0,
                dim=0,
            )
        elif self.subsample_type == SubsampleType.exponential_plus_latest:
            probs = normalize(
                exponential * self.subsample_params["weight"]
                + latest * (1.0 - self.subsample_params["weight"]),
                p=1.0,
                dim=0,
            )
        elif self.subsample_type == SubsampleType.uniform_plus_latest:
            probs = normalize(
                uniform * self.subsample_params["weight"]
                + latest * (1.0 - self.subsample_params["weight"]),
                p=1.0,
                dim=0,
            )
        else:
            raise ValueError(f'invalid subsample_type "{self.subsample_type}"')
        return dist.Categorical(probs).sample([n])

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
        loss = self.svi.step(self.xs, self.ws, self.subsample())
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
