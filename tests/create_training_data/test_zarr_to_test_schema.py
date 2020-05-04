import numpy as np
from zarr_to_test_schema import (
    sample,
    Domain,
    Range,
    Array,
    ChunkedArray,
    MyEncoder,
    CoordinateSchema,
    VariableSchema,
    DatasetSchema,
    dumps,
    loads
)
import pickle

import pytest


def test_sample_middle_dim():
    arr = np.ones((3, 4, 5))
    expected = sample(arr, sample_axes=[1], num_samples=2)
    assert expected.shape == (2, 3, 5)


def test_sample_multiple_sample_axes():
    arr = np.ones((3, 4, 5))
    expected = sample(arr, sample_axes=[1, 2], num_samples=2)
    assert expected.shape == (2, 3)


def test_sample_multiple_sample_axes_reorder():
    arr = np.ones((4, 5, 3))
    expected = sample(arr, sample_axes=[0, 1], num_samples=2)
    assert expected.shape == (2, 3)


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
    e = MyEncoder()
    assert e.encode(np.dtype("float32")) == '"' + np.dtype("float32").str + '"'


def test_encoder_array():
    e = MyEncoder()
    assert e.encode(np.array([1, 2, 3])) == "[1, 2, 3]"


def test_DatasetSchema_dumps():

    x = CoordinateSchema("x", ["x"], np.array([1, 2, 3]))
    a = VariableSchema(
        "a",
        ["x"],
        ChunkedArray(shape=[3], chunks=[1], dtype=np.dtype("float32")),
        domain=Range(0, 10),
    )

    ds = DatasetSchema(coords=[x], variables=[a])

    val  = dumps(ds)
    assert isinstance(val, str)


def test_DatasetSchema_dumps_regression(regtest):
    x = CoordinateSchema("x", ["x"], np.array([1, 2, 3]))
    a = VariableSchema(
        "a",
        ["x"],
        ChunkedArray(shape=[3], chunks=[1], dtype=np.dtype("float32")),
        domain=Range(0, 10),
    )

    ds = DatasetSchema(coords=[x], variables=[a])

    val  = dumps(ds)
    print(val, file=regtest)


def test_DatasetSchemaLoads():
    encoded_data = """
    {"coords": [{"name": "x", "dims": ["x"], "value": [1, 2, 3]}], "variables": [{"name": "a", "dims": ["x"], "array": {"shape": [3], "dtype": "<f4", "chunks": [1]}, "domain": {"min": 0, "max": 10}}]}
    """ # noqa

    ds = loads(encoded_data)
    assert isinstance(ds, DatasetSchema)

    v = ds.variables[0]
    assert isinstance(v, VariableSchema)
    assert isinstance(v.domain, Domain)


def test_generate_and_pickle_integration():
    x = CoordinateSchema("x", ["x"], np.array([1, 2, 3]))
    a = VariableSchema(
        "a",
        ["x"],
        ChunkedArray(shape=[3], chunks=[1], dtype=np.dtype("float32")),
        domain=Range(0, 10),
    )

    ds = DatasetSchema(coords=[x], variables=[a])
    d = ds.generate()
    pickle.dumps(d)


def test_generate_regression(regtest):
    x = CoordinateSchema("x", ["x"], np.array([1, 2, 3]))
    a = VariableSchema(
        "a",
        ["x"],
        ChunkedArray(shape=[3], chunks=[1], dtype=np.dtype("float32")),
        domain=Range(0, 10),
    )

    ds = DatasetSchema(coords=[x], variables=[a])
    d = ds.generate()
    arr = d.a.values
    print(arr, file=regtest)