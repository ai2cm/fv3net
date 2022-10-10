import dacite
from dataclasses import dataclass
from typing import Optional


@dataclass
class ReservoirHyperparameters:
    """Hyperparameters for reservoir

    input_dim: Size of input vector
    reservoir_state_dim: Size of hidden state vector,
        W_res has shape reservoir_state_dim x reservoir_state_dim
    sparsity: Fraction of elements in W_res that are zero
    output_dim: Size of output vector. Can be smaller than input
        dimension if predicting on subdomains with overlapping
        input regions.
    spectral_radius: Largest absolute value eigenvalue of W_res.
        Larger values increase the memory of the reservoir.
    seed: Random seed for sampling
    input_coupling_sparsity: Fraction of elements in each row of W_in
        that are zero. Kept the same in all rows to ensure each input
        is equally connected into the reservoir. Defaults to 0.
    input_coupling_scaling: Scaling applied to W_in. Defaults to 1,
        where all elements are sampled from random uniform distribution
        [-1, 1]. Changing this affects relative weighting of reservoir memory
        versus the most recent state.
    res_scaling: Optional scale value for W_res that can be provided in lieu of
        spectral radius. This is useful if you know what scaling parameter
        applied to the uniform distribution [0, 1] for a given reservoir size
        will lead to the (approximate) desired spectral radius, since eigenvalue
        calculation for larger reservoirs can be slow.
    """

    input_dim: int
    reservoir_state_dim: int
    sparsity: float
    output_dim: Optional[int] = None

    spectral_radius: Optional[float] = None

    seed: int = 0
    input_coupling_sparsity: float = 0.0
    input_coupling_scaling: float = 1.0
    res_scaling: Optional[float] = None

    def __post_init__(self):
        if self.spectral_radius and self.res_scaling:
            raise ValueError("Only one of spectral_radius or scaling can be specified")
        if not self.output_dim:
            self.output_dim = self.input_dim


@dataclass
class ReadoutHyperparameters:
    l2: float
    square_half_hidden_state: bool = False


@dataclass
class ReservoirTrainingConfig:
    reservoir_hyperparameters: ReservoirHyperparameters
    readout_hyperparameters: ReadoutHyperparameters
    n_burn: int
    noise: float
    seed: int = 0
    n_samples: Optional[int] = None
    n_jobs: int = -1
    subdomain_output_size: Optional[int] = None
    subdomain_overlap: Optional[int] = None
    subdomain_axis: int = 1

    @classmethod
    def from_dict(cls, kwargs) -> "ReservoirTrainingConfig":
        kwargs = {**kwargs}  # make a copy to avoid mutating the input
        # custom enums must be specified for dacite to handle correctly
        dacite_config = dacite.Config(strict=True, cast=[bool, str, int, float])
        kwargs["reservoir_hyperparameters"] = dacite.from_dict(
            data_class=ReservoirHyperparameters,
            data=kwargs.get("reservoir_hyperparameters", {}),
            config=dacite_config,
        )
        kwargs["readout_hyperparameters"] = dacite.from_dict(
            data_class=ReadoutHyperparameters,
            data=kwargs.get("readout_hyperparameters", {}),
            config=dacite_config,
        )
        return dacite.from_dict(
            data_class=ReservoirTrainingConfig,
            data=kwargs,
            config=dacite.Config(strict=True),
        )
