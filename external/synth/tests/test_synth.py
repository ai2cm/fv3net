import pickle

import numpy as np
import zarr

from synth import (
    read_schema_from_zarr,
    Array,
    ChunkedArray,
    CoordinateSchema,
    DatasetSchema,
    Domain,
    MyEncoder,
    Range,
    VariableSchema,
    __version__,
    dumps,
    loads,
)


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

    val = dumps(ds)
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

    val = dumps(ds)
    print(val, file=regtest)


def test_DatasetSchemaLoads():
    encoded_data = """
    {"coords": [{"name": "x", "dims": ["x"], "value": [1, 2, 3], "attrs": [1]}], "variables": [{"name": "a", "dims": ["x"], "array": {"shape": [3], "dtype": "<f4", "chunks": [1]}, "domain": {"min": 0, "max": 10}}]}
    """  # noqa

    ds = loads(encoded_data)
    assert isinstance(ds, DatasetSchema)

    v = ds.variables[0]
    assert isinstance(v, VariableSchema)
    assert isinstance(v.domain, Domain)

    assert ds.coords[0].attrs == [1]


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
    ds = schema.generate()

    assert dict(ds.time.attrs) == dict(julian_time_attrs)
