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
from .domain2 import RankXYDivider
from fv3fit._shared import io
from .utils import square_even_terms
from .transformers import ReloadableTransfomer, encode_columns, decode_columns


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
        rank_divider: RankXYDivider,
        autoencoder: ReloadableTransfomer,
        square_half_hidden_state: bool = False,
    ):
        # TODO: The autoencoder and by  extension the rank encoder all assume
        # that the same variable set are used for inputs, hybrid variables, and
        # perhaps outputs.  This will quickly not be the case, so need to allow
        # for different ones for each case.  Default can be an expectation that
        # all variables are available, but the encoder will then contain all that
        # information (e.g., time=t and t+1).  Separate PR is necessary for that.
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

        self._no_overlap_divider = self.rank_divider.get_no_overlap_rank_divider()

    def predict(self, hybrid_input: Sequence[np.ndarray]):
        # hybrid input is assumed to be in original spatial xy dims
        # (x, y, feature) and does not include overlaps.
        encoded_hybrid_input = encode_columns(
            input_arrs=hybrid_input, transformer=self.autoencoder
        )

        flat_hybrid_in = self._no_overlap_divider.get_all_subdomains_with_flat_feature(
            encoded_hybrid_input
        )
        readout_input = self._concatenate_readout_inputs(
            self.reservoir_model.reservoir.state, flat_hybrid_in
        )

        flat_prediction = self.readout.predict(readout_input)
        prediction = self._no_overlap_divider.merge_all_flat_feature_subdomains(
            flat_prediction
        )
        decoded_prediction = decode_columns(
            encoded_output=prediction, transformer=self.autoencoder,
        )
        return decoded_prediction

    def _concatenate_readout_inputs(self, hidden_state_input, flat_hybrid_input):
        # TODO: square even terms should be handled by Reservoir model?
        if self.square_half_hidden_state is True:
            hidden_state_input = square_even_terms(hidden_state_input, axis=-1)

        readout_input = np.concatenate([hidden_state_input, flat_hybrid_input], axis=-1)
        return readout_input

    def reset_state(self):
        self.reservoir_model.reset_state()

    def increment_state(self, prediction_with_overlap: Sequence[np.ndarray]) -> None:
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
        self.model.increment_state(xy_input_arrs)

    def reset_state(self):
        self.model.reset_state()

    def _input_dataset_to_arrays(self, inputs: xr.Dataset) -> Sequence[np.ndarray]:
        # Converts from xr dataset to sequence of variable ndarrays expected by encoder
        # Make sure the xy dimensions match the rank divider
        transposed_inputs = _transpose_xy_dims(
            ds=inputs, rank_dims=self.model.rank_divider._rank_dims_all_features
        )
        input_arrs = []
        for variable in self.model.input_variables:
            da = transposed_inputs[variable]
            if "z" not in da.dims:
                da = da.expand_dims("z", axis=-1)
            input_arrs.append(da.values)
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
        rank_divider: RankXYDivider,
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

        self._no_overlap_divider = rank_divider.get_no_overlap_rank_divider()

    def process_state_to_readout_input(self):
        readout_input = self.reservoir.state
        if self.square_half_hidden_state is True:
            readout_input = square_even_terms(readout_input, axis=0)

        return readout_input

    def predict(self):
        # Returns raw readout prediction of latent state.
        readout_input = self.process_state_to_readout_input()
        flat_prediction = self.readout.predict(readout_input)
        prediction = self._no_overlap_divider.merge_all_flat_feature_subdomains(
            flat_prediction
        )
        decoded_prediction = decode_columns(
            encoded_output=prediction, transformer=self.autoencoder,
        )
        return decoded_prediction

    def reset_state(self):
        input_shape = (
            self.rank_divider.n_subdomains,
            self.reservoir.hyperparameters.state_size,
        )
        self.reservoir.reset_state(input_shape)

    def increment_state(self, prediction_with_overlap: Sequence[np.ndarray]) -> None:
        # input array is in native x, y, z_feature coordinates
        encoded_xy_input_arrs = encode_columns(
            prediction_with_overlap, self.autoencoder
        )
        encoded_flat_sub = self.rank_divider.get_all_subdomains_with_flat_feature(
            encoded_xy_input_arrs
        )
        self.reservoir.increment_state(encoded_flat_sub)

    def synchronize(self, synchronization_time_series):
        # input arrays in native x, y, z_feature coordinates
        encoded_timeseries = encode_columns(
            synchronization_time_series, self.autoencoder
        )
        encoded_flat = self.rank_divider.get_all_subdomains_with_flat_feature(
            encoded_timeseries
        )
        self.reservoir.synchronize(encoded_flat)

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

        rank_divider = RankXYDivider.load(os.path.join(path, cls._RANK_DIVIDER_NAME))

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
