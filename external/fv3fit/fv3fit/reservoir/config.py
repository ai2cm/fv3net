import dacite
from dataclasses import dataclass, asdict
from typing import Sequence, Tuple, Optional, Set
import fsspec
import yaml
from .._shared.training_config import Hyperparameters


@dataclass
class CubedsphereSubdomainConfig:
    layout: Tuple[int, int]
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
class TransformerConfig:
    input: Optional[str] = None
    output: Optional[str] = None
    hybrid: Optional[str] = None


@dataclass
class ReservoirTrainingConfig(Hyperparameters):
    """
    input_variables: variables and additional features in time series
    output_variables: time series variables, must be subset of input_variables
    reservoir_hyperparameters: hyperparameters for reservoir
    readout_hyperparameters: hyperparameters for readout
    n_timesteps_synchronize: number of timesteps at start of time series to use
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
    transformers: optional TransformerConfig for autoencoders to use in
        encoding input, output, and/or hybrid variable sets.
    mask_reservoir_inputs: Apply the mask_variable to the reservoir inputs (not the
        hybrid, which is specified by mask_hybrid_inputs). Mask is applied to the input
        array before multiplication with W_in.
    mask_hybrid_inputs: Apply the mask_variable to the hybrid inputs.  Applied prior
        to input to the readout layer.
    mask_variable: mask variable to use if mask_reservoir_inputs or mask_readout
        are specified, or to use for validation if validate_sst_only is specified.
    validate_sst_only: if True, perform the SST validation instead of the atmosphere
        validation code
    """

    input_variables: Sequence[str]
    output_variables: Sequence[str]
    subdomain: CubedsphereSubdomainConfig
    reservoir_hyperparameters: ReservoirHyperparameters
    readout_hyperparameters: BatchLinearRegressorHyperparameters
    n_timesteps_synchronize: int
    input_noise: float
    seed: int = 0
    transformers: Optional[TransformerConfig] = None
    n_jobs: Optional[int] = 1
    square_half_hidden_state: bool = False
    hybrid_variables: Optional[Sequence[str]] = None
    mask_reservoir_inputs: bool = False
    mask_hybrid_inputs: bool = False
    mask_variable: Optional[str] = None
    validate_sst_only: bool = False
    _METADATA_NAME = "reservoir_training_config.yaml"

    def __post_init__(self):
        if self.hybrid_variables is not None:
            hybrid_and_input_vars_intersection = set(
                self.hybrid_variables
            ).intersection(self.input_variables)
            if len(hybrid_and_input_vars_intersection) > 0:
                raise ValueError(
                    f"Hybrid variables {self.hybrid_variables} cannot overlap with "
                    f"input variables {self.input_variables}."
                )
        if (
            self.mask_reservoir_inputs or self.mask_hybrid_inputs
        ) and self.mask_variable is None:
            raise ValueError("mask_variable must be specified if masking is enabled")

    @property
    def variables(self) -> Set[str]:
        if self.hybrid_variables is not None:
            additional_vars = list(self.hybrid_variables)  # type: ignore
        else:
            additional_vars = []
        if self.mask_variable is not None:
            additional_vars.append(self.mask_variable)
        return set(
            list(self.input_variables) + list(self.output_variables) + additional_vars
        )

    @classmethod
    def from_dict(cls, kwargs) -> "ReservoirTrainingConfig":
        kwargs = {**kwargs}
        dacite_config = dacite.Config(strict=True, cast=[bool, str, int, float, tuple])
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
        kwargs["transformers"] = dacite.from_dict(
            data_class=TransformerConfig,
            data=kwargs.get("transformers", {}),
            config=dacite_config,
        )
        return dacite.from_dict(
            data_class=ReservoirTrainingConfig,
            data=kwargs,
            config=dacite.Config(strict=True),
        )

    def dump(self, path: str):
        metadata = {
            "n_timesteps_synchronize": self.n_timesteps_synchronize,
            "input_noise": self.input_noise,
            "seed": self.seed,
            "n_jobs": self.n_jobs,
            "reservoir_hyperparameters": asdict(self.reservoir_hyperparameters),
            "readout_hyperparameters": asdict(self.readout_hyperparameters),
            "subdomain": asdict(self.subdomain),
            "transformers": asdict(self.transformers),
            "input_variables": self.input_variables,
            "output_variables": self.output_variables,
            "hybrid_variables": self.hybrid_variables,
            "mask_reservoir_inputs": self.mask_reservoir_inputs,
            "mask_hybrid_inputs": self.mask_hybrid_inputs,
            "mask_variable": self.mask_variable,
            "validate_sst_only": self.validate_sst_only,
        }
        fs: fsspec.AbstractFileSystem = fsspec.get_fs_token_paths(path)[0]
        fs.makedirs(path, exist_ok=True)
        mapper = fs.get_mapper(path)
        mapper[self._METADATA_NAME] = yaml.safe_dump(metadata, indent=4).encode("UTF-8")
