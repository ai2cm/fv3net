from __future__ import annotations
import numpy as np
import os
import typing
from typing import Iterable, Hashable, Sequence, Union, Mapping
import xarray as xr

import fv3fit
from fv3fit import Predictor
from fv3fit._shared import io
from .model import (
    HybridReservoirComputingModel,
    ReservoirComputingModel,
    ReservoirModelType,
)


def _transpose_ordered_dims(ds_dims: Sequence[str], rank_dims: Sequence[str]):
    # Useful for transposing the x, y, z dims in a dataset to match those in
    # RankDivider.rank_dims, and leaves other dims in the same order
    # relative to x,y. Dims after the first occurence of one of the rank_dims
    # are assumed to be feature dims.
    # e.g. (time, y, x, z) -> (time, x, y, z) for rank_dims=(x, y)
    leading_non_xyz_dims = []
    rank_dims_in_data = [dim for dim in rank_dims if dim in ds_dims]
    for dim in ds_dims:
        if dim not in rank_dims_in_data:
            leading_non_xyz_dims.append(dim)
        if dim in rank_dims_in_data:
            break
    ordered_dims = (*leading_non_xyz_dims, *rank_dims_in_data)
    return ordered_dims


class DatasetAdapter:
    DIM_ORDER = ["x", "y", "z"]

    def __init__(
        self, input_variables: Iterable[Hashable], output_variables: Iterable[Hashable],
    ):
        self.input_variables = input_variables
        self.output_variables = output_variables

    def _ndarray_to_dataarray(self, arr: np.ndarray) -> xr.DataArray:
        if len(arr.shape) == 3:
            if arr.shape[-1] == 1:
                arr = arr[:, :, 0]
                dims = self.DIM_ORDER[:2]
            else:
                dims = self.DIM_ORDER
        elif len(arr.shape) == 2:
            dims = self.DIM_ORDER[:2]
        else:
            raise (ValueError(f"Array must have 2 or 3 dims, got {arr.shape}"))
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

        transposed_input_dims = _transpose_ordered_dims(
            ds_dims=list(inputs.dims), rank_dims=self.DIM_ORDER
        )
        transposed_inputs = inputs.transpose(*transposed_input_dims)
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
        self.nonhybrid_input_variables = model.input_variables
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

    def get_model_from_subdomain(self, subdomain_index: int) -> ReservoirDatasetAdapter:
        model = self.model.get_model_from_subdomain(subdomain_index)
        return ReservoirDatasetAdapter(
            model=model,
            input_variables=model.input_variables,
            output_variables=model.output_variables,
        )

    def dump(self, path):
        self.model.dump(os.path.join(path, self.MODEL_DIR))

    @classmethod
    def load(cls, path: str) -> "ReservoirDatasetAdapter":
        model = ReservoirComputingModel.load(os.path.join(path, cls.MODEL_DIR))
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
        self.nonhybrid_input_variables = model.input_variables
        self.hybrid_variables = model.hybrid_variables
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

    def get_model_from_subdomain(
        self, subdomain_index: int
    ) -> HybridReservoirDatasetAdapter:
        model = self.model.get_model_from_subdomain(subdomain_index)
        return HybridReservoirDatasetAdapter(
            model=model,
            input_variables=model.input_variables,
            output_variables=model.output_variables,
        )

    def dump(self, path):
        self.model.dump(os.path.join(path, self.MODEL_DIR))

    @classmethod
    def load(cls, path: str) -> "HybridReservoirDatasetAdapter":
        model = HybridReservoirComputingModel.load(os.path.join(path, cls.MODEL_DIR))
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
    if isinstance(model, ReservoirDatasetAdapter) or isinstance(
        model, HybridReservoirDatasetAdapter
    ):
        divider = model.model.rank_divider
    else:
        divider = model.rank_divider

    return [model.get_model_from_subdomain(i) for i in range(divider.n_subdomains)]


def generate_subdomain_models_from_saved_model(model_path, output_path, model_index=0):
    """
    Generate a set of subdomain models from a saved model and save them to
    a directory.

    model_path: path to a save model (remote paths supported)
    output_path: path to save subdomain models to (remote paths supported)
    model_index: index of the model to save (default 0)  used as a starting index
        to number each subdomain model.
    """
    model = fv3fit.load(model_path)
    split_models = split_multi_subdomain_model(model)
    submodel_map = {}
    for i, to_save in enumerate(split_models, start=model_index * len(split_models)):
        submodel_output_path = os.path.join(output_path, f"subdomain_{i}")
        submodel_map[i] = submodel_output_path
        fv3fit.dump(to_save, submodel_output_path)

    return submodel_map


def generate_subdomain_models_from_model_map(
    model_map: Mapping[int, str], output_path: str
) -> Mapping[int, str]:
    """
    Generate a set of subdomain models from each model in the model map.

    model_map: mapping from a model index to a path of the saved model
    output_path: path to save subdomain models to (remote paths supported)
    """
    submodel_map = {}
    for tile_index, model_path in model_map.items():
        submodel_map.update(
            generate_subdomain_models_from_saved_model(
                model_path, output_path, model_index=tile_index
            )
        )

    return submodel_map
