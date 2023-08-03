import fsspec
import numpy as np
import os
from typing import Iterable, Hashable, Sequence, cast, Union, TypeVar
import xarray as xr
import yaml

import fv3fit
from fv3fit import Predictor
from .domain import RankDivider
from fv3fit._shared import io
from .utils import square_even_terms
from .model import HybridReservoirComputingModel, ReservoirComputingModel
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


ReservoirModel = TypeVar("T", HybridReservoirComputingModel, ReservoirComputingModel)


@io.register("reservoir-adapter")
class ReservoirDatasetAdapter(Predictor):
    def __init__(
        self,
        input_variables: Iterable[Hashable],
        output_variables: Iterable[Hashable],
        model: ReservoirModel,
    ) -> None:
        self.model = model
        if isinstance(model, HybridReservoirComputingModel):
            self.input_variables = list(
                set(input_variables.union(model.hybrid_variables))
            )
        else:
            self.input_variables = input_variables
        self.output_variables = output_variables

    def _ndarray_to_dataarray(self, arr: np.ndarray) -> xr.DataArray:
        dims = [*self.model.rank_divider.rank_dims]
        if len(arr.shape) == 3 and arr.shape[-1] > 1:
            dims.append("z")
        return xr.DataArray(data=arr, dims=dims)

    def _output_array_to_ds(
        self, outputs: Sequence[np.ndarray], output_dims: Sequence[str]
    ) -> xr.Dataset:
        return xr.Dataset(
            {
                var: self._ndarray_to_dataarray(output)
                for var, output in zip(self.model.output_variables, outputs)
            }
        ).transpose(*output_dims)

    def predict(self, inputs: xr.Dataset) -> xr.Dataset:
        # TODO: potentially use in train.py instead of special functions there
        xy_input_arrs = self._input_dataset_to_arrays(inputs)  # x, y, feature dims
        predict_args = []
        if isinstance(self.model, HybridReservoirComputingModel):
            predict_args = [
                xy_input_arrs,
            ]
        prediction_arr = self.model.predict(**predict_args)
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
            ds=inputs, rank_dims=self.model.rank_divider.rank_dims
        )
        input_arrs = []
        for variable in self.model.input_variables:
            da = transposed_inputs[variable]
            if "z" not in da.dims:
                da = da.expand_dims("z", axis=-1)
            input_arrs.append(da.values)
        return input_arrs
