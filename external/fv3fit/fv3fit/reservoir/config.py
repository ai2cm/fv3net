import dacite
from dataclasses import dataclass, asdict
from typing import Sequence, Optional, Set
import fsspec
import yaml
from .._shared.training_config import Hyperparameters


@dataclass
class CubedsphereSubdomainConfig:
    layout: Sequence[int]
    overlap: int
    rank_dims: Sequence[str]


@dataclass
class ReservoirHyperparameters:
    """Hyperparameters for reservoir

    state_size: Size of hidden state vector,
        W_res has shape state_size x state_size
    adjacency_matrix_sparsity: Fraction of elements in adjacency matrix
        W_res that are zero
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

    state_size: int
    adjacency_matrix_sparsity: float
    spectral_radius: float
    seed: int = 0
    input_coupling_sparsity: float = 0.0
    input_coupling_scaling: float = 1.0


@dataclass
class BatchLinearRegressorHyperparameters:
    """
    l2: ridge regression coefficient
    add_bias_term: Use default of True if input samples do not already
        have a constant term to fit the intercept. Default True value is
        the same behavior as sklearn regressors.
    use_least_squares_solve: Can set to True for simple test cases
        where the system is underdetermined and the default np.linalg.solve
        encounters errors with singular XT.X
    """

    l2: float
    add_bias_term: bool = True
    use_least_squares_solve: bool = False


@dataclass
class ReservoirTrainingConfig(Hyperparameters):
    """
    input_variables: variables and additional features in time series
    output_variables: time series variables, must be subset of input_variables
    reservoir_hyperparameters: hyperparameters for reservoir
    readout_hyperparameters: hyperparameters for readout
    n_batches_burn: number of training batches at start of time series to use
        for synchronizaton.  This data is  used to update the reservoir state
        but is not included in training.
    input_noise: stddev of normal distribution which is sampled to add input
        noise to the training inputs when generating hidden states. This is
        commonly done to aid in the stability of the RC model.
    seed: random seed for sampling
    subdomain: Subdomain config. All subdomains use the same reservoir weights;
        one readout is created and trained for each subdomain. Subdomain size
        and reservoir input size much match.
    square_half_hidden_state: if True, square even terms in the reservoir
        state before it is used as input to the regressor's .fit and
        .predict methods. This option was found to be important for skillful
        predictions in Wikner+2020 (https://doi.org/10.1063/5.0005541)
    autoencoder_path: optional path for autoencoder to use in encoding time series
        before passing to reservoir
    """

    input_variables: Sequence[str]
    output_variables: Sequence[str]
    subdomain: CubedsphereSubdomainConfig
    reservoir_hyperparameters: ReservoirHyperparameters
    readout_hyperparameters: BatchLinearRegressorHyperparameters
    n_batches_burn: int
    input_noise: float
    seed: int = 0
    n_jobs: Optional[int] = 1
    square_half_hidden_state: bool = False
    autoencoder_path: Optional[str] = None
    hybrid_variables: Optional[Sequence[str]] = None
    _METADATA_NAME = "reservoir_training_config.yaml"

    def __post_init__(self):
        if set(self.output_variables).issubset(self.input_variables) is False:
            raise ValueError(
                f"Output variables {self.output_variables} must be a subset of "
                f"input variables {self.input_variables}."
            )
        if self.hybrid_variables is not None:
            hybrid_and_input_vars_intersection = set(
                self.hybrid_variables
            ).intersection(self.input_variables)
            if len(hybrid_and_input_vars_intersection) > 0:
                raise ValueError(
                    f"Hybrid variables {self.hybrid_variables} cannot overlap with "
                    f"input variables {self.input_variables}."
                )

    @property
    def variables(self) -> Set[str]:
        hybrid_vars = list(self.hybrid_variables) or []  # type: ignore
        return set(list(self.input_variables) + hybrid_vars)

    @classmethod
    def from_dict(cls, kwargs) -> "ReservoirTrainingConfig":
        kwargs = {**kwargs}
        dacite_config = dacite.Config(strict=True, cast=[bool, str, int, float])
        kwargs["reservoir_hyperparameters"] = dacite.from_dict(
            data_class=ReservoirHyperparameters,
            data=kwargs.get("reservoir_hyperparameters", {}),
            config=dacite_config,
        )
        kwargs["readout_hyperparameters"] = dacite.from_dict(
            data_class=BatchLinearRegressorHyperparameters,
            data=kwargs.get("readout_hyperparameters", {}),
            config=dacite_config,
        )
        kwargs["subdomain"] = dacite.from_dict(
            data_class=CubedsphereSubdomainConfig,
            data=kwargs.get("subdomain", {}),
            config=dacite_config,
        )
        return dacite.from_dict(
            data_class=ReservoirTrainingConfig,
            data=kwargs,
            config=dacite.Config(strict=True),
        )

    def dump(self, path: str):
        metadata = {
            "n_batches_burn": self.n_batches_burn,
            "input_noise": self.input_noise,
            "seed": self.seed,
            "n_jobs": self.n_jobs,
            "reservoir_hyperparameters": asdict(self.reservoir_hyperparameters),
            "readout_hyperparameters": asdict(self.readout_hyperparameters),
            "subdomain": asdict(self.subdomain),
            "autoencoder_path": self.autoencoder_path,
        }
        fs: fsspec.AbstractFileSystem = fsspec.get_fs_token_paths(path)[0]
        fs.makedirs(path, exist_ok=True)
        mapper = fs.get_mapper(path)
        mapper[self._METADATA_NAME] = yaml.safe_dump(metadata, indent=4).encode("UTF-8")
