"""
Concepts:

1. sampler (needs to be efficient)
    Some dimensions have NaNs (e.g. surface fields with step='begin' step='output')
2. random variable (trained from sample)
    Single variate or multivariate (along a height dimension for instance)
3. output-description


chunks: (we need to be to generate chunked data)


"""
from typing import Sequence, Tuple, Mapping
import json
from dataclasses import dataclass, asdict, field
import zarr
import numpy as np
import dask.array as da
import xarray as xr
import logging
import io

logger = logging.getLogger(__file__)


# TODO can probably delete
def sample_xr(variable: xr.DataArray, vary_dims=(), num_samples=100) -> xr.DataArray:
    """Sample a variable along vary_dims (stacking is non-ideal since it combines chunks)
    """
    sample_dims = list(set(variable.dims) - set(vary_dims))
    return (
        variable.stack(feature=vary_dims, sample=sample_dims)
        .isel(sample=slice(0, num_samples))
        .transpose("feature", "sample")
    )


# TODO can probably delete
def fit_xr(rv, sample: xr.DataArray) -> xr.DataArray:
    arr = np.asarray(sample)
    stats = np.apply_along_axis(rv.fit, sample)
    return xr.DataArray(stats, dims=["feature"], coords={"feature": sample.feature})


# TODO can probably delete
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


# TODO rename
class MyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        try:
            return o.str
        except AttributeError:
            pass
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, o)


class Domain:
    # TODO don't use this double dispatch mechanism for generating
    # It seems a little over-engineered at this point
    def generate_chunked(self, shape, chunks, dtype):
        raise NotImplementedError

    def generate_array(self, shape, dtype):
        raise NotImplementedError


@dataclass
class Range(Domain):
    min: float
    max: float

    def _generate_like(self, x):
        return self.generate_array(x.shape, x.dtype)

    def generate_chunked(self, shape, chunks, dtype) -> da.Array:
        darr = da.empty(shape, chunks=chunks, dtype=dtype)
        return darr.map_blocks(self._generate_like)

    def generate_array(self, shape, dtype) -> np.ndarray:
        # need to set random seed here
        # to make this generation deterministic
        np.random.seed(0)
        return np.random.uniform(low=self.min, high=self.max, size=shape).astype(dtype)


@dataclass
class Array:
    shape: Sequence[int]
    dtype: np.dtype

    def generate(self, domain: Domain):
        return domain.generate_array(self.shape, self.dtype)


@dataclass
class ChunkedArray:
    shape: Sequence[int]
    dtype: np.dtype
    chunks: Tuple[Tuple[int]]

    # TODO probably remove these generating functions
    # seems a poor separation of concerns.
    def generate(self, domain: Domain):
        return domain.generate_chunked(self.shape, self.chunks, self.dtype)


@dataclass
class VariableSchema:
    name: str
    dims: Sequence[str]
    # TODO cast array to a list or tuple to avoid serializing numpy arrays
    array: Array

    # TODO probably want to decouple "Domain" from the schema object
    domain: Domain

    attrs: Mapping = field(default_factory=dict)

    def generate(self):
        return xr.DataArray(
            self.array.generate(self.domain), dims=self.dims, name=self.name
            ,attrs=self.attrs,
        )


@dataclass
class CoordinateSchema:
    name: str
    dims: Sequence[str]
    value: np.ndarray
    attrs: Mapping = field(default_factory=dict)

    def generate(self):
        return xr.DataArray(self.value, dims=self.dims, name=self.name, attrs=self.attrs)


@dataclass
class DatasetSchema:
    coords: Sequence[CoordinateSchema]
    variables: Sequence[VariableSchema]

    def generate(self):
        return xr.Dataset(
            {v.name: v.generate() for v in self.variables},
            coords={v.name: v.generate() for v in self.coords},
        )


# TODO test this function
def read_schema_from_zarr(
    group,
    default_range=Range(-1000, 1000),
    sample=lambda arr: arr[-1, 0, 0, 0],
    coords=("forecast_time", "initial_time", "tile", "step", "z", "y", "x"),
):

    variables = []
    coord_schemes = []

    for variable in group:
        logger.info(f"Reading {variable}")
        arr = group[variable]
        attrs = dict(arr.attrs)

        dims = attrs.pop("_ARRAY_DIMENSIONS")

        if variable in coords:
            scheme = CoordinateSchema(variable, [variable], arr[:], attrs)
            coord_schemes.append(scheme)
        else:
            n = sample(arr)

            # TODO arr.chunks is a not of ints, so the properties of
            # chunked array are not accurate
            array = ChunkedArray(arr.shape, arr.dtype, arr.chunks)

            m, M = n.min(), n.max()
            if np.isnan(m) or np.isnan(M):
                range_ = default_range
            else:
                range_ = Range(m.item(), M.item())

            scheme = VariableSchema(
                variable, dims, array, domain=range_,
                attrs=attrs,
            )
            variables.append(scheme)

    return DatasetSchema(coord_schemes, variables)


def dump(schema: DatasetSchema, fp):
    json.dump(asdict(schema), fp, cls=MyEncoder)


def dumps(schema: DatasetSchema):
    fp = io.StringIO()
    dump(schema, fp)
    return fp.getvalue()


def load(fp):
    d = json.load(fp)

    coords = []
    for coord in d["coords"]:
        coords.append(
            CoordinateSchema(
                **coord
            )
        )

    variables = []
    for variable in d["variables"]:
        array = ChunkedArray(**variable.pop("array"))
        # TODO this should work with any subtype of Domain
        # Maybe add an attribute to the encoder? Or maybe this is over-engineering,
        # and we can only use Range
        range_ = Range(**variable.pop("domain"))
        variables.append(VariableSchema(array=array, domain=range_, **variable))

    return DatasetSchema(coords=coords, variables=variables)


def loads(s):
    fp = io.StringIO(s)
    return load(fp)
