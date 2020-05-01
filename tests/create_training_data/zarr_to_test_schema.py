"""
Concepts:

1. sampler (needs to be efficient)
    Some dimensions have NaNs (e.g. surface fields with step='begin' step='output')
2. random variable (trained from sample)
    Single variate or multivariate (along a height dimension for instance)
3. output-description


chunks: (we need to be to generate chunked data)


"""
from typing import Sequence, Tuple
import fsspec
from dataclasses import dataclass
import zarr
from scipy.stats import uniform
import dask.array as da
import numpy as np
import xarray as xr
from toolz import first


def sample_xr(variable: xr.DataArray, vary_dims=(), num_samples=100) -> xr.DataArray:
    """Sample a variable along vary_dims (stacking is non-ideal since it combines chunks)
    """
    sample_dims = list(set(variable.dims) - set(vary_dims))
    return (
        variable.stack(feature=vary_dims, sample=sample_dims)
        .isel(sample=slice(0, num_samples))
        .transpose("feature", "sample")
    )


def fit_xr(rv, sample: xr.DataArray) -> xr.DataArray:
    arr = np.asarray(sample)
    stats = np.apply_along_axis(rv.fit, sample)
    return xr.DataArray(stats, dims=["feature"], coords={"feature": sample.feature})


def sample(variable: zarr.Array, sample_axes=(), num_samples=100) -> np.ndarray:
    """Sample a variable along the first axis listed in sample_axes

    Samples are enumerated along the first axis of the resulting numpy array
    """

    sample_axes = list(sample_axes)

    idx = slice(0, 10)
    sample_axis = min(sample_axes)
    axes = {}
    for i in range(variable.ndim):
        if i in sample_axes:
            axes[i] = slice(num_samples) if i == sample_axis else 0
        else:
            axes[i] = slice(None)

    idx = list(axes[i] for i in range(len(axes)))
    arr = variable[idx]
    return np.moveaxis(arr, sample_axis, 0)


class Domain:
    def generate_chunked(self, shape, chunks, dtype):
        raise NotImplementedError

    def generate_array(self, shape, dtype):
        raise NotImplementedError


@dataclass
class Range(Domain):
    min: float
    max: float

    def generate_chunked(self, shape, chunks, dtype) -> da.Array:
        darr = da.empty(shape, chunks=chunks, dtype=dtype)

        def func(x):
            return self.generate_array(x.shape, x.dtype)

        return darr.map_blocks(func)

    def generate_array(self, shape, dtype) -> np.ndarray:
        return np.random.uniform(low=self.min, high=self.max, size=shape).astype(dtype)


@dataclass
class Array:
    shape: Sequence[int]
    dtype: np.dtype

    def generate(self, domain: Domain):
        return domain.generate_array(self.shape, self.dtype)


@dataclass
class ChunkedArray(Array):
    shape: Sequence[int]
    dtype: np.dtype
    chunks: Tuple[Tuple[int]]

    def generate(self, domain: Domain):
        return domain.generate_chunked(self.shape, self.chunks, self.dtype)


@dataclass
class VariableSchema:
    name: str
    dims: Sequence[str]
    array: Array
    domain: Domain

    def generate(self):
        return xr.DataArray(
            self.array.generate(self.domain), dims=self.dims, name=self.name
        )


@dataclass
class CoordinateSchema:
    name: str
    dims: Sequence[str]
    value: np.ndarray

    def generate(self):
        return xr.DataArray(self.value, dims=self.dims, name=self.name)


@dataclass
class DatasetSchema:
    coords: Sequence[CoordinateSchema]
    variables: Sequence[VariableSchema]

    def generate(self):
        return xr.Dataset(
            {v.name: v.generate() for v in self.variables},
            coords={v.name: v.generate() for v in self.coords},
        )



if __name__ == "__main__":
    url = "gs://vcm-ml-data/test-end-to-end-integration/integration-debug/one_step_run_/big.zarr"

    mapper = fsspec.get_mapper(url)
    group = zarr.open_group(mapper)
    rv = {}

    coords = ["forecast_time", "initial_time", "tile", "step", "z", "y", "x"]

    variables = []
    coord_schemes = []

    for variable in group:
        arr = group[variable]

        if variable in coords:
            scheme = CoordinateSchema(variable, [variable], arr.shape, arr[:])
            coord_schemes.append(scheme)
        else:
            n = arr[-1, 0, 0, 0]

            scheme = VariableSchema(
                variable,
                arr.attrs["_ARRAY_DIMENSIONS"],
                arr.shape,
                Range(dtype=arr.dtype, min=n.min(), max=n.max()),
            )
            variables.append(scheme)

    schema = DatasetSchema(coord_schemes, variables)
    print(schema)
