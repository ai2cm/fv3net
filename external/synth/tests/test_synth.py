import pickle

import numpy as np
import zarr

import pytest

from synth.core import _Encoder

from synth import (
    generate,
    read_schema_from_zarr,
    Array,
    ChunkedArray,
    CoordinateSchema,
    DatasetSchema,
    Range,
    VariableSchema,
    __version__,
    dumps,
)

from synth.core import dict_to_schema


def test_version():
    assert __version__ == "0.1.0"


def test_generate_array():
    r = Range(0, 1)
    a = Array(shape=(1, 2), dtype=np.float32)
    out = a.generate(r)

    assert out.dtype == a.dtype
    assert out.shape == a.shape


def test_generate_chunked_array():
    r = Range(0, 1)
    a = ChunkedArray(shape=(10, 10), dtype=np.float32, chunks=((5, 5), (5, 5)))
    out = a.generate(r)

    assert out.dtype == a.dtype
    assert out.shape == a.shape
    assert out.chunks == a.chunks


def test_encoder_dtype():
    e = _Encoder()
    assert e.encode(np.dtype("float32")) == '"' + np.dtype("float32").str + '"'


def test_encoder_array():
    e = _Encoder()
    assert e.encode(np.array([1, 2, 3])) == "[1, 2, 3]"


def test_DatasetSchema_dumps():

    x = CoordinateSchema("x", ["x"], np.array([1, 2, 3]))
    a = VariableSchema(
        "a", ["x"], ChunkedArray(shape=[3], chunks=[1], dtype=np.dtype("float32")),
    )
    ds = DatasetSchema(coords=[x], variables=[a])

    val = dumps(ds)
    assert isinstance(val, str)


def test_DatasetSchema_dumps_regression(regtest):
    x = CoordinateSchema("x", ["x"], np.array([1, 2, 3]))
    a = VariableSchema(
        "a", ["x"], ChunkedArray(shape=[3], chunks=[1], dtype=np.dtype("float32")),
    )

    ds = DatasetSchema(coords=[x], variables=[a])

    val = dumps(ds)
    print(val, file=regtest)


mock_dict_schema_v1 = {
    "coords": [{"name": "x", "dims": ["x"], "value": [1, 2, 3], "attrs": [1]}],
    "variables": [
        {
            "name": "a",
            "dims": ["x"],
            "array": {"shape": [3], "dtype": "<f4", "chunks": [1]},
            "range": {"min": 0, "max": 10},
        }
    ],
}

mock_dict_schema_v2alpha = {
    "version": "v2alpha",
    "schema": {
        "coords": [{"name": "x", "dims": ["x"], "value": [1, 2, 3], "attrs": [1]}],
        "variables": [
            {
                "name": "a",
                "dims": ["x"],
                "array": {"shape": [3], "dtype": "<f4", "chunks": [1]},
            }
        ],
    },
}


@pytest.mark.parametrize("d", [mock_dict_schema_v1, mock_dict_schema_v2alpha])
def test_DatasetSchemaLoads(d):
    ds = dict_to_schema(d)
    assert isinstance(ds, DatasetSchema)

    v = ds.variables[0]
    assert isinstance(v, VariableSchema)
    assert ds.coords[0].attrs == [1]


def test_generate_and_pickle_integration():
    x = CoordinateSchema("x", ["x"], np.array([1, 2, 3]))
    a = VariableSchema(
        "a", ["x"], ChunkedArray(shape=[3], chunks=[1], dtype=np.dtype("float32"))
    )

    ranges = {"a": Range(0, 10)}

    ds = DatasetSchema(coords=[x], variables=[a])
    d = generate(ds, ranges)
    pickle.dumps(d)


def test_generate_regression(regtest):
    """
    Note:

        regtest fixture provided by pytest-regtest plugin:
        https://pypi.org/project/pytest-regtest/

    """
    x = CoordinateSchema("x", ["x"], np.array([1, 2, 3]))
    a = VariableSchema(
        "a", ["x"], ChunkedArray(shape=[3], chunks=[1], dtype=np.dtype("float32")),
    )

    ranges = {"a": Range(0, 10)}

    ds = DatasetSchema(coords=[x], variables=[a])
    d = generate(ds, ranges)
    arr = d.a.values
    print(arr, file=regtest)


def test_cftime_generate():
    julian_time_attrs = {
        "calendar": "julian",
        "calendar_type": "JULIAN",
        "cartesian_axis": "T",
        "long_name": "time",
        "units": "seconds since 2016-08-01T00:15:00.000026",
    }

    store = {}

    group = zarr.open_group(store, mode="w")
    arr = group.zeros("time", shape=[1], dtype=np.float64)
    arr.attrs.update(julian_time_attrs)
    arr.attrs["_ARRAY_DIMENSIONS"] = ["time"]

    schema = read_schema_from_zarr(group, coords=["time"])
    ds = generate(schema, {})

    assert dict(ds.time.attrs) == dict(julian_time_attrs)
