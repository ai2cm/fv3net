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

SCHEMA_VERSION = "v1"


class _Encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        try:
            return o.str
        except AttributeError:
            pass
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, o)


@dataclass
class Range:
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

    def generate(self, range: Range):
        return range.generate_array(self.shape, self.dtype)


@dataclass
class ChunkedArray:
    shape: Sequence[int]
    dtype: np.dtype
    chunks: Tuple[Tuple[int]]

    # TODO probably remove these generating functions
    # seems a poor separation of concerns.
    def generate(self, range: Range):
        return range.generate_chunked(self.shape, self.chunks, self.dtype)


@dataclass
class VariableSchema:
    name: str
    dims: Sequence[str]
    array: Array
    range: Range

    attrs: Mapping = field(default_factory=dict)

    def generate(self):
        return xr.DataArray(
            self.array.generate(self.range),
            dims=self.dims,
            name=self.name,
            attrs=self.attrs,
        )


@dataclass
class CoordinateSchema:
    name: str
    dims: Sequence[str]
    value: np.ndarray
    attrs: Mapping = field(default_factory=dict)

    def generate(self):
        return xr.DataArray(
            self.value, dims=self.dims, name=self.name, attrs=self.attrs
        )


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
    group: zarr.Group,
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

            scheme = VariableSchema(variable, dims, array, range=range_, attrs=attrs,)
            variables.append(scheme)

    return DatasetSchema(coord_schemes, variables)


def dump(schema: DatasetSchema, fp):
    output = {"version": SCHEMA_VERSION, "schema": asdict(schema)}
    json.dump(output, fp, cls=_Encoder)


def dumps(schema: DatasetSchema):
    fp = io.StringIO()
    dump(schema, fp)
    return fp.getvalue()


# To bump the version make a new parser named 
# dict_to_schema_<version>
def dict_to_schema_v1(d):

    coords = []
    for coord in d["coords"]:
        coords.append(CoordinateSchema(**coord))

    variables = []
    for variable in d["variables"]:
        array = ChunkedArray(**variable.pop("array"))
        range_ = Range(**variable.pop("range"))
        variables.append(VariableSchema(array=array, range=range_, **variable))

    return DatasetSchema(coords=coords, variables=variables)


def load(fp):
    d = json.load(fp)

    try:
        version = d["version"]
        schema = d["schema"]
    except KeyError:
        version = "v1"
        schema = d

    loader = globals()["dict_to_schema_" + version]
    return loader(schema)


def loads(s):
    fp = io.StringIO(s)
    return load(fp)
