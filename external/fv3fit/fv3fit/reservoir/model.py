import fsspec
import numpy as np
import os
from typing import Optional, Iterable, Hashable
import yaml

from fv3fit import Predictor
from .readout import ReservoirComputingReadout
from .reservoir import Reservoir
from .domain import RankDivider
from fv3fit._shared import io
from .utils import square_even_terms
from .transformers.autoencoder import Autoencoder
from ._reshaping import flatten_2d_keeping_columns_contiguous


@io.register("hybrid-reservoir")
class HybridReservoirComputingModel(Predictor):
    _HYBRID_VARIABLES_NAME = "hybrid_variables.yaml"

    def __init__(
        self,
        input_variables: Iterable[Hashable],
        hybrid_variables: Iterable[Hashable],
        output_variables: Iterable[Hashable],
        reservoir: Reservoir,
        readout: ReservoirComputingReadout,
        square_half_hidden_state: bool = False,
        rank_divider: Optional[RankDivider] = None,
        autoencoder: Optional[Autoencoder] = None,
    ):
        self.reservoir_model = ReservoirComputingModel(
            input_variables=input_variables,
            output_variables=output_variables,
            reservoir=reservoir,
            readout=readout,
            square_half_hidden_state=square_half_hidden_state,
            rank_divider=rank_divider,
            autoencoder=autoencoder,
        )
        self.input_variables = input_variables
        self.hybrid_variables = hybrid_variables
        self.output_variables = output_variables
        self.readout = readout
        self.square_half_hidden_state = square_half_hidden_state
        self.rank_divider = rank_divider
        self.autoencoder = autoencoder

    def predict(self, hybrid_input):
        readout_input_from_reservoir = (
            self.reservoir_model.process_state_to_readout_input()
        )
        readout_input = np.concatenate([readout_input_from_reservoir, hybrid_input])
        prediction = self.readout.predict(readout_input).reshape(-1)
        return prediction

    def reset_state(self):
        self.reservoir_model.reset_state()

    def increment_state(self):
        self.reservoir_model.increment_state()

    def synchronize(self, synchronization_time_series):
        self.reservoir_model.synchronize(synchronization_time_series)

    def dump(self, path: str) -> None:
        self.reservoir_model.dump(path)
        with fsspec.open(os.path.join(path, self._HYBRID_VARIABLES_NAME), "w") as f:
            f.write(yaml.dump({"hybrid_variables": self.hybrid_variables}))

    @classmethod
    def load(cls, path: str) -> "HybridReservoirComputingModel":
        pure_reservoir_model = ReservoirComputingModel.load(path)
        with fsspec.open(os.path.join(path, cls._HYBRID_VARIABLES_NAME), "r") as f:
            hybrid_variables = yaml.safe_load(f)["hybrid_variables"]
        return cls(
            input_variables=pure_reservoir_model.input_variables,
            output_variables=pure_reservoir_model.output_variables,
            reservoir=pure_reservoir_model.reservoir,
            readout=pure_reservoir_model.readout,
            square_half_hidden_state=pure_reservoir_model.square_half_hidden_state,
            rank_divider=pure_reservoir_model.rank_divider,
            autoencoder=pure_reservoir_model.autoencoder,
            hybrid_variables=hybrid_variables,
        )


@io.register("pure-reservoir")
class ReservoirComputingModel(Predictor):
    _RESERVOIR_SUBDIR = "reservoir"
    _READOUT_SUBDIR = "readout"
    _METADATA_NAME = "metadata.yaml"
    _RANK_DIVIDER_NAME = "rank_divider.yaml"
    _AUTOENCODER_SUBDIR = "autoencoder"

    def __init__(
        self,
        input_variables: Iterable[Hashable],
        output_variables: Iterable[Hashable],
        reservoir: Reservoir,
        readout: ReservoirComputingReadout,
        square_half_hidden_state: bool = False,
        rank_divider: Optional[RankDivider] = None,
        autoencoder: Optional[Autoencoder] = None,
    ):
        """_summary_

        Args:
            reservoir: Reservoir which takes input and updates hidden state
            readout: readout layer which takes in state and predicts next time step
            square_half_hidden_state: if True, square even terms in the reservoir
                state before it is used as input to the regressor's .fit and
                .predict methods. This option was found to be important for skillful
                predictions in Wikner+2020 (https://doi.org/10.1063/5.0005541).
            rank_divider: object used to divide and reconstruct domain <-> subdomains
        """
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.reservoir = reservoir
        self.readout = readout
        self.square_half_hidden_state = square_half_hidden_state
        self.rank_divider = rank_divider
        self.autoencoder = autoencoder

    def process_state_to_readout_input(self):
        if self.square_half_hidden_state is True:
            readout_input = square_even_terms(self.reservoir.state, axis=0)
        else:
            readout_input = self.reservoir.state
        # For prediction over multiple subdomains (>1 column in reservoir state
        # array), flatten state into 1D vector before predicting
        readout_input = flatten_2d_keeping_columns_contiguous(readout_input)
        return readout_input

    def predict(self):
        # Returns raw readout prediction of latent state.
        readout_input = self.process_state_to_readout_input()
        prediction = self.readout.predict(readout_input).reshape(-1)
        return prediction

    def reset_state(self):
        if self.rank_divider is not None:
            input_shape = (
                self.reservoir.hyperparameters.state_size,
                self.rank_divider.n_subdomains,
            )
        else:
            input_shape = (self.reservoir.hyperparameters.state_size,)
        self.reservoir.reset_state(input_shape)

    def increment_state(self, prediction_with_overlap):
        self.reservoir.increment_state(prediction_with_overlap)

    def synchronize(self, synchronization_time_series):
        self.reservoir.synchronize(synchronization_time_series)

    def dump(self, path: str) -> None:
        """Dump data to a directory

        Args:
            path: a URL pointing to a directory
        """
        self.reservoir.dump(os.path.join(path, self._RESERVOIR_SUBDIR))
        self.readout.dump(os.path.join(path, self._READOUT_SUBDIR))

        metadata = {
            "square_half_hidden_state": self.square_half_hidden_state,
            "input_variables": self.input_variables,
            "output_variables": self.output_variables,
        }
        with fsspec.open(os.path.join(path, self._METADATA_NAME), "w") as f:
            f.write(yaml.dump(metadata))

        if self.rank_divider is not None:
            self.rank_divider.dump(os.path.join(path, self._RANK_DIVIDER_NAME))
        if self.autoencoder is not None:
            self.autoencoder.dump(os.path.join(path, self._AUTOENCODER_SUBDIR))

    @classmethod
    def load(cls, path: str) -> "ReservoirComputingModel":
        """Load a model from a remote path"""
        reservoir = Reservoir.load(os.path.join(path, cls._RESERVOIR_SUBDIR))
        readout = ReservoirComputingReadout.load(
            os.path.join(path, cls._READOUT_SUBDIR)
        )
        with fsspec.open(os.path.join(path, cls._METADATA_NAME), "r") as f:
            metadata = yaml.safe_load(f)

        fs: fsspec.AbstractFileSystem = fsspec.get_fs_token_paths(path)[0]

        if fs.exists(os.path.join(path, cls._RANK_DIVIDER_NAME)):
            rank_divider = RankDivider.load(os.path.join(path, cls._RANK_DIVIDER_NAME))
        else:
            rank_divider = None

        if fs.exists(os.path.join(path, cls._AUTOENCODER_SUBDIR)):
            autoencoder = Autoencoder.load(os.path.join(path, cls._AUTOENCODER_SUBDIR))
        else:
            autoencoder = None  # type: ignore
        return cls(
            input_variables=metadata["input_variables"],
            output_variables=metadata["output_variables"],
            reservoir=reservoir,
            readout=readout,
            square_half_hidden_state=metadata["square_half_hidden_state"],
            rank_divider=rank_divider,
            autoencoder=autoencoder,
        )
