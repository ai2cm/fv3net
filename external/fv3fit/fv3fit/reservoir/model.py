import fsspec
import numpy as np
import os
from typing import Iterable, Hashable, Sequence, cast
import xarray as xr
import yaml

import fv3fit
from fv3fit import Predictor
from .readout import ReservoirComputingReadout
from .reservoir import Reservoir
from .domain import RankDivider
from fv3fit._shared import io
from .utils import square_even_terms
from .transformers import ReloadableTransfomer, encode_columns, decode_columns
from ._reshaping import flatten_2d_keeping_columns_contiguous


def _transpose_xy_dims(ds: xr.Dataset, rank_dims: Sequence[str]):
    # Useful for transposing the x, y dims in a dataset to match those in
    # RankDivider.rank_dims, and leaves other dims in the same order
    # relative to x,y. Dims after the first occurence of one of the rank_dims
    # are assumed to be feature dims.
    # e.g. (time, y, x, z) -> (time, x, y, z) for rank_dims=(x, y)
    leading_non_xy_dims = []
    for dim in ds.dims:
        if dim not in rank_dims:
            leading_non_xy_dims.append(dim)
        if dim in rank_dims:
            break
    ordered_dims = (*leading_non_xy_dims, *rank_dims)
    return ds.transpose(*ordered_dims, ...)


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
        rank_divider: RankDivider,
        autoencoder: ReloadableTransfomer,
        square_half_hidden_state: bool = False,
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

    def predict(self, hybrid_input: Sequence[np.ndarray]):
        # hybrid input is assumed to be in original spatial xy dims
        # (x, y, feature) and does not include overlaps.
        encoded_hybrid_input = encode_columns(
            input_arrs=hybrid_input, transformer=self.autoencoder
        )
        flat_encoded_hybrid_input = self.rank_divider.flatten_subdomains_to_columns(
            encoded_hybrid_input, with_overlap=False
        )
        flattened_readout_input = self._concatenate_readout_inputs(
            self.reservoir_model.reservoir.state, flat_encoded_hybrid_input
        )
        flat_prediction = self.readout.predict(flattened_readout_input).reshape(-1)
        prediction = self.rank_divider.merge_subdomains(flat_prediction)
        decoded_prediction = decode_columns(
            encoded_output=prediction,
            transformer=self.autoencoder,
            xy_shape=self.rank_divider.rank_extent_without_overlap,
        )
        return decoded_prediction

    def _concatenate_readout_inputs(self, hidden_state_input, flat_hybrid_input):
        # hybrid input is flattened prior to being input to this step
        if self.square_half_hidden_state is True:
            hidden_state_input = square_even_terms(hidden_state_input, axis=0)

        readout_input = np.concatenate([hidden_state_input, flat_hybrid_input], axis=0)
        flattened_readout_input = flatten_2d_keeping_columns_contiguous(readout_input)
        return flattened_readout_input

    def reset_state(self):
        self.reservoir_model.reset_state()

    def increment_state(self, prediction_with_overlap):
        self.reservoir_model.increment_state(prediction_with_overlap)

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


class HybridDatasetAdapter:
    def __init__(self, model: HybridReservoirComputingModel) -> None:
        self.model = model

    def predict(self, inputs: xr.Dataset) -> xr.Dataset:
        # TODO: potentially use in train.py instead of special functions there
        xy_input_arrs = self._input_dataset_to_arrays(inputs)  # x, y, feature dims

        prediction_arr = self.model.predict(xy_input_arrs)
        return self._output_array_to_ds(prediction_arr, dims=list(inputs.dims))

    def increment_state(self, inputs: xr.Dataset):
        xy_input_arrs = self._input_dataset_to_arrays(inputs)  # x, y, feature dims
        encoded_xy_input_arrs = encode_columns(xy_input_arrs, self.model.autoencoder)
        subdomains = self.model.rank_divider.flatten_subdomains_to_columns(
            encoded_xy_input_arrs, with_overlap=True
        )
        self.model.increment_state(subdomains)

    def reset_state(self):
        self.model.reset_state()

    def _input_dataset_to_arrays(self, inputs: xr.Dataset) -> Sequence[np.ndarray]:
        # Converts from xr dataset to sequence of variable ndarrays expected by encoder
        # Make sure the xy dimensions match the rank divider
        transposed_inputs = _transpose_xy_dims(
            ds=inputs, rank_dims=self.model.rank_divider.rank_dims
        )
        input_arrs = [
            transposed_inputs[variable].values
            for variable in self.model.input_variables
        ]
        return input_arrs

    def _output_array_to_ds(
        self, outputs: Sequence[np.ndarray], dims: Sequence[str]
    ) -> xr.Dataset:
        ds = xr.Dataset(
            {
                var: (dims, outputs[i])
                for i, var in enumerate(self.model.output_variables)
            }
        )
        return ds


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
        rank_divider: RankDivider,
        autoencoder: ReloadableTransfomer,
        square_half_hidden_state: bool = False,
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
        flat_prediction = self.readout.predict(readout_input).reshape(-1)
        prediction = self.rank_divider.merge_subdomains(flat_prediction)
        decoded_prediction = decode_columns(
            encoded_output=prediction,
            transformer=self.autoencoder,
            xy_shape=self.rank_divider.rank_extent_without_overlap,
        )
        return decoded_prediction

    def reset_state(self):
        input_shape = (
            self.reservoir.hyperparameters.state_size,
            self.rank_divider.n_subdomains,
        )
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

        self.rank_divider.dump(os.path.join(path, self._RANK_DIVIDER_NAME))
        if self.autoencoder is not None:
            fv3fit.dump(self.autoencoder, os.path.join(path, self._AUTOENCODER_SUBDIR))

    @classmethod
    def load(cls, path: str) -> "ReservoirComputingModel":
        """Load a model from a remote path"""
        reservoir = Reservoir.load(os.path.join(path, cls._RESERVOIR_SUBDIR))
        readout = ReservoirComputingReadout.load(
            os.path.join(path, cls._READOUT_SUBDIR)
        )
        with fsspec.open(os.path.join(path, cls._METADATA_NAME), "r") as f:
            metadata = yaml.safe_load(f)

        rank_divider = RankDivider.load(os.path.join(path, cls._RANK_DIVIDER_NAME))

        autoencoder = cast(
            ReloadableTransfomer,
            fv3fit.load(os.path.join(path, cls._AUTOENCODER_SUBDIR)),
        )
        return cls(
            input_variables=metadata["input_variables"],
            output_variables=metadata["output_variables"],
            reservoir=reservoir,
            readout=readout,
            square_half_hidden_state=metadata["square_half_hidden_state"],
            rank_divider=rank_divider,
            autoencoder=autoencoder,
        )
