import numpy as np
from typing import Iterable, Hashable, Sequence
import xarray as xr

from fv3fit import Predictor
from fv3fit._shared import io
from .model import HybridReservoirComputingModel, ReservoirComputingModel


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


class DatasetAdapter:
    def __init__(
        self,
        input_variables: Iterable[Hashable],
        output_variables: Iterable[Hashable],
        rank_dims: Sequence[str],
    ):
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.rank_dims = rank_dims

    def _ndarray_to_dataarray(self, arr: np.ndarray) -> xr.DataArray:
        dims = [*self.rank_dims]
        if len(arr.shape) == 3 and arr.shape[-1] > 1:
            dims.append("z")
        return xr.DataArray(data=arr, dims=dims)

    def output_array_to_ds(
        self, outputs: Sequence[np.ndarray], output_dims: Sequence[str]
    ) -> xr.Dataset:
        return xr.Dataset(
            {
                var: self._ndarray_to_dataarray(output)
                for var, output in zip(self.output_variables, outputs)
            }
        ).transpose(*output_dims)

    def input_dataset_to_arrays(
        self, inputs: xr.Dataset, variables: Iterable[Hashable]
    ) -> Sequence[np.ndarray]:
        # Converts from xr dataset to sequence of variable ndarrays expected by encoder
        # Make sure the xy dimensions match the rank divider
        transposed_inputs = _transpose_xy_dims(ds=inputs, rank_dims=self.rank_dims)
        input_arrs = []
        for variable in variables:
            da = transposed_inputs[variable]
            if "z" not in da.dims:
                da = da.expand_dims("z", axis=-1)
            input_arrs.append(da.values)
        return input_arrs


@io.register("reservoir-adapter")
class ReservoirDatasetAdapter(Predictor):
    def __init__(
        self,
        model: ReservoirComputingModel,
        input_variables: Iterable[Hashable],
        output_variables: Iterable[Hashable],
    ) -> None:
        self.model = model
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.model_adapter = DatasetAdapter(
            input_variables=input_variables,
            output_variables=output_variables,
            rank_dims=model.rank_divider.rank_dims,
        )

    def predict(self, inputs: xr.Dataset) -> xr.Dataset:
        # inputs arg is not used, but is required by Predictor signature and prog run
        prediction_arr = self.model.predict()
        return self.model_adapter.output_array_to_ds(
            prediction_arr, output_dims=list(inputs.dims)
        )

    def increment_state(self, inputs: xr.Dataset):
        xy_input_arrs = self.model_adapter.input_dataset_to_arrays(
            inputs, self.input_variables
        )  # x, y, feature dims
        self.model.increment_state(xy_input_arrs)

    def reset_state(self):
        self.model.reset_state()

    def dump(self, path):
        self.model.dump(path)

    @classmethod
    def load(cls, path: str) -> "ReservoirDatasetAdapter":
        model = ReservoirComputingModel.load(path)
        model.reset_state()
        adapter = cls(
            input_variables=model.input_variables,
            output_variables=model.output_variables,
            model=model,
        )
        return adapter


@io.register("hybrid-reservoir-adapter")
class HybridReservoirDatasetAdapter(Predictor):
    def __init__(
        self,
        model: HybridReservoirComputingModel,
        input_variables: Iterable[Hashable],
        output_variables: Iterable[Hashable],
    ) -> None:
        self.model = model
        self.input_variables = list(set(input_variables).union(model.hybrid_variables))
        self.output_variables = output_variables
        self.model_adapter = DatasetAdapter(
            input_variables=self.input_variables,
            output_variables=output_variables,
            rank_dims=model.rank_divider.rank_dims,
        )

    def predict(self, inputs: xr.Dataset) -> xr.Dataset:
        xy_input_arrs = self.model_adapter.input_dataset_to_arrays(
            inputs, self.model.hybrid_variables
        )  # x, y, feature dims

        prediction_arr = self.model.predict(xy_input_arrs)
        return self.model_adapter.output_array_to_ds(
            prediction_arr, output_dims=list(inputs.dims)
        )

    def increment_state(self, inputs: xr.Dataset):
        xy_input_arrs = self.model_adapter.input_dataset_to_arrays(
            inputs, self.input_variables
        )  # x, y, feature dims
        self.model.increment_state(xy_input_arrs)

    def reset_state(self):
        self.model.reset_state()

    def dump(self, path):
        self.model.dump(path)

    @classmethod
    def load(cls, path: str) -> "HybridReservoirDatasetAdapter":
        model = HybridReservoirComputingModel.load(path)
        model.reset_state()
        adapter = cls(
            input_variables=model.input_variables,
            output_variables=model.output_variables,
            model=model,
        )
        return adapter
