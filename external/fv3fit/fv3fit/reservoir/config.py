import dacite
from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class ReservoirHyperparameters:
    """Hyperparameters for reservoir

    input_size: Size of input vector
    state_size: Size of hidden state vector,
        W_res has shape state_size x state_size
    adjacency_matrix_sparsity: Fraction of elements in adjacency matrix
        W_res that are zero
    output_size: Optional: size of output vector. Can be smaller than input
        dimension if predicting on subdomains with overlapping
        input regions. Defaults to same as input_size
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
    """

    input_size: int
    state_size: int
    adjacency_matrix_sparsity: float
    spectral_radius: float
    output_size: Optional[int] = None
    seed: int = 0
    input_coupling_sparsity: float = 0.0
    input_coupling_scaling: float = 1.0

    def __post_init__(self):
        if not self.output_size:
            self.output_size = self.input_size


@dataclass
class ReadoutHyperparameters:
    """
    l2: Ridge regression coefficient for the linear regression
    solver: solver to use for sklearn Ridge regressor
    square_half_hidden_state: if True, square even elements of state vector
        as described in in Wikner+ 2020 (https://doi.org/10.1063/5.0005541)
    """

    l2: float
    solver: Literal[
        "auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"
    ] = "auto"
    square_half_hidden_state: bool = False


@dataclass
class ReservoirTrainingConfig:
    reservoir_hyperparameters: ReservoirHyperparameters
    readout_hyperparameters: ReadoutHyperparameters
    n_burn: int
    input_noise: float
    seed: int = 0
    n_samples: Optional[int] = None
    n_jobs: int = -1
    subdomain_output_size: Optional[int] = None
    subdomain_overlap_size: Optional[int] = None
    subdomain_axis: int = 1

    """_summary_

    Returns:
        _type_: _description_
    """

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
