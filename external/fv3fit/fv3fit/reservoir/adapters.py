from copy import copy
import numpy as np
import os
import typing
from typing import Iterable, Hashable, Sequence, Union
import xarray as xr

import fv3fit
from fv3fit import Predictor
from fv3fit._shared import io
from .model import (
    HybridReservoirComputingModel,
    ReservoirComputingModel,
    ReservoirModelType,
)
from .domain2 import RankXYDivider


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
    DIM_ORDER_2D = ["x", "y"]

    def __init__(
        self, input_variables: Iterable[Hashable], output_variables: Iterable[Hashable],
    ):
        self.input_variables = input_variables
        self.output_variables = output_variables

    def _ndarray_to_dataarray(self, arr: np.ndarray) -> xr.DataArray:
        dims = [*self.DIM_ORDER_2D]
        if len(arr.shape) == 3:
            if arr.shape[-1] > 1:
                dims.append("z")
            elif arr.shape[-1] == 1:
                arr = arr[:, :, 0]
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
        transposed_inputs = _transpose_xy_dims(ds=inputs, rank_dims=self.DIM_ORDER_2D)
        input_arrs = []
        for variable in variables:
            da = transposed_inputs[variable]
            if "z" not in da.dims:
                da = da.expand_dims("z", axis=-1)
            input_arrs.append(da.values)
        return input_arrs


@io.register("reservoir-adapter")
class ReservoirDatasetAdapter(Predictor):
    MODEL_DIR = "reservoir_model"

    def __init__(
        self,
        model: ReservoirComputingModel,
        input_variables: Iterable[Hashable],
        output_variables: Iterable[Hashable],
    ) -> None:
        """Wraps a reservoir model to take in and return xarray datasets.
        The initialization args for input and output variables are not used and
        are included for matching the signature of the Predictor parent class.
        The input and output variables are set using the model arg's input and
        output variable sets.
        """
        self.model = model
        self.input_variables = model.input_variables
        self.output_variables = model.output_variables
        self.model_adapter = DatasetAdapter(
            input_variables=self.input_variables,
            output_variables=self.output_variables,
        )

    @property
    def input_overlap(self):
        """Number of halo points expected for reservoir increment inputs"""
        return self.model.rank_divider.overlap

    @property
    def is_hybrid(self):
        return False

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
        self.model.dump(os.path.join(path, self.MODEL_DIR))

    @classmethod
    def load(cls, path: str) -> "ReservoirDatasetAdapter":
        model = ReservoirComputingModel.load(os.path.join(path, cls.MODEL_DIR))
        model.reset_state()
        adapter = cls(
            input_variables=model.input_variables,
            output_variables=model.output_variables,
            model=model,
        )
        return adapter


@io.register("hybrid-reservoir-adapter")
class HybridReservoirDatasetAdapter(Predictor):
    MODEL_DIR = "hybrid_reservoir_model"

    def __init__(
        self,
        model: HybridReservoirComputingModel,
        input_variables: Iterable[Hashable],
        output_variables: Iterable[Hashable],
    ) -> None:
        """Wraps a hybrid reservoir model to take in and return xarray datasets.
        The initialization args for input and output variables are not used and
        are included for matching the signature of the Predictor parent class.
        The input and output variables are set using the model arg's input, output,
        and hybrid variable sets.
        """
        self.model = model
        self.input_variables = list(
            set(model.input_variables).union(model.hybrid_variables)
        )
        self.output_variables = model.output_variables
        self.model_adapter = DatasetAdapter(
            input_variables=self.input_variables,
            output_variables=model.output_variables,
        )

    @property
    def input_overlap(self):
        """Number of halo points expected for reservoir increment inputs"""
        return self.model.rank_divider.overlap

    @property
    def is_hybrid(self):
        return True

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
            inputs, self.model.input_variables
        )  # x, y, feature dims
        self.model.increment_state(xy_input_arrs)

    def reset_state(self):
        self.model.reset_state()

    def dump(self, path):
        self.model.dump(os.path.join(path, self.MODEL_DIR))

    @classmethod
    def load(cls, path: str) -> "HybridReservoirDatasetAdapter":
        model = HybridReservoirComputingModel.load(os.path.join(path, cls.MODEL_DIR))
        model.reset_state()
        adapter = cls(
            input_variables=model.input_variables,
            output_variables=model.output_variables,
            model=model,
        )
        return adapter


ReservoirAdapterType = Union[ReservoirDatasetAdapter, HybridReservoirDatasetAdapter]
ReservoirModelLike = Union[ReservoirModelType, ReservoirAdapterType]


@typing.no_type_check
def split_multi_subdomain_model(
    model: ReservoirModelLike,
) -> Sequence[ReservoirModelLike]:
    """ Split a multi-subdomain model into a list of single subdomain models.
    """
    is_adapter = isinstance(model, ReservoirDatasetAdapter) or isinstance(
        model, HybridReservoirDatasetAdapter
    )
    if is_adapter:
        adapter = model
        model = adapter.model
        is_adapter = True
    else:
        adapter = None
        is_adapter = False

    rank_divider = model.rank_divider
    readout = model.readout

    if rank_divider.n_subdomains == 1:
        raise ValueError("Model must have multiple subdomains to split.")

    new_rank_divider = RankXYDivider(
        subdomain_layout=(1, 1),
        overlap=rank_divider.overlap,
        overlap_rank_extent=rank_divider.subdomain_extent,
        z_feature_size=rank_divider._z_feature_size,
    )

    new_kwargs = {
        "input_variables": model.input_variables,
        "output_variables": model.output_variables,
        "reservoir": model.reservoir,
        "transformers": model.transformers,
        "square_half_hidden_state": model.square_half_hidden_state,
    }

    if isinstance(model, HybridReservoirComputingModel):
        new_kwargs["hybrid_variables"] = model.hybrid_variables

    new_models = []
    for i in range(rank_divider.n_subdomains):
        new_readout = readout.get_subdomain_readout(i)
        new_model = model.__class__(
            readout=new_readout, rank_divider=copy(new_rank_divider), **new_kwargs
        )

        if is_adapter and adapter is not None:
            if adapter.is_hybrid:
                _adapter_cls = HybridReservoirDatasetAdapter
            else:
                _adapter_cls = ReservoirDatasetAdapter

            new_adapter = _adapter_cls(
                new_model,
                input_variables=model.input_variables,
                output_variables=model.output_variables,
            )
            new_model = new_adapter

        new_models.append(new_model)

    return new_models


def generate_subdomain_models_for_tile(model_path, output_path, tile_index=0):
    model = fv3fit.load(model_path)
    split_models = split_multi_subdomain_model(model)
    for i, to_save in enumerate(split_models, start=tile_index * len(split_models)):
        fv3fit.dump(to_save, os.path.join(output_path, f"subdomain_{i}"))


def generate_subdomain_models_from_all_tiles(tile_model_map, output_path):
    for tile_index, model_path in tile_model_map.items():
        generate_subdomain_models_for_tile(
            model_path, output_path, tile_index=tile_index
        )
