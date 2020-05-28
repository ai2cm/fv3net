import vcm
import synth
from synth import DatasetSchema, CoordinateSchema, ChunkedArray, VariableSchema
import numpy as np
import xarray as xr


def fv_core_schema(n: int, nz: int, x, xi, y, yi, z):

    return DatasetSchema(
        coords={
            x: CoordinateSchema(
                name=x,
                dims=[x],
                value=np.arange(n),
                attrs={"long_name": x, "units": "none", "cartesian_axis": "X"},
            ),
            xi: CoordinateSchema(
                name=xi,
                dims=[xi],
                value=np.arange(n + 1),
                attrs={"long_name": xi, "units": "none", "cartesian_axis": "X"},
            ),
            yi: CoordinateSchema(
                name=yi,
                dims=[yi],
                value=np.arange(n + 1),
                attrs={"long_name": yi, "units": "none", "cartesian_axis": "Y"},
            ),
            y: CoordinateSchema(
                name=y,
                dims=[y],
                value=np.arange(n),
                attrs={"long_name": y, "units": "none", "cartesian_axis": "Y"},
            ),
            z: CoordinateSchema(
                name=z,
                dims=[z],
                value=np.arange(nz),
                attrs={"long_name": z, "units": "none", "cartesian_axis": "Z"},
            ),
            "Time": CoordinateSchema(
                name="Time",
                dims=["Time"],
                value=np.array([1.0], dtype=np.float32),
                attrs={
                    "long_name": "Time",
                    "units": "time level",
                    "cartesian_axis": "T",
                },
            ),
        },
        variables={
            "u": VariableSchema(
                name="u",
                dims=["Time", z, yi, x],
                array=ChunkedArray(
                    shape=(1, nz, n + 1, n),
                    dtype=np.dtype("float32"),
                    chunks=(1, nz, n + 1, n),
                ),
                attrs={
                    "long_name": "u",
                    "units": "none",
                    "checksum": "61E9EC149EA76D64",
                },
            ),
            "v": VariableSchema(
                name="v",
                dims=["Time", z, y, xi],
                array=ChunkedArray(
                    shape=(1, nz, n, n + 1),
                    dtype=np.dtype("float32"),
                    chunks=(1, nz, n, n + 1),
                ),
                attrs={
                    "long_name": "v",
                    "units": "none",
                    "checksum": "EB861B394529209B",
                },
            ),
            "W": VariableSchema(
                name="W",
                dims=["Time", z, y, x],
                array=ChunkedArray(
                    shape=(1, nz, n, n),
                    dtype=np.dtype("float32"),
                    chunks=(1, nz, n, n),
                ),
                attrs={
                    "long_name": "W",
                    "units": "none",
                    "checksum": "9238BEBCAD91DA31",
                },
            ),
            "DZ": VariableSchema(
                name="DZ",
                dims=["Time", z, y, x],
                array=ChunkedArray(
                    shape=(1, nz, n, n),
                    dtype=np.dtype("float32"),
                    chunks=(1, nz, n, n),
                ),
                attrs={
                    "long_name": "DZ",
                    "units": "none",
                    "checksum": "A0E2B6B530BBD16F",
                },
            ),
            "T": VariableSchema(
                name="T",
                dims=["Time", z, y, x],
                array=ChunkedArray(
                    shape=(1, nz, n, n),
                    dtype=np.dtype("float32"),
                    chunks=(1, nz, n, n),
                ),
                attrs={
                    "long_name": "T",
                    "units": "none",
                    "checksum": "1CE4FC9FF8237F91",
                },
            ),
            "delp": VariableSchema(
                name="delp",
                dims=["Time", z, y, x],
                array=ChunkedArray(
                    shape=(1, nz, n, n),
                    dtype=np.dtype("float32"),
                    chunks=(1, nz, n, n),
                ),
                attrs={
                    "long_name": "delp",
                    "units": "none",
                    "checksum": "674B269C950EF9E1",
                },
            ),
            "phis": VariableSchema(
                name="phis",
                dims=["Time", y, x],
                array=ChunkedArray(
                    shape=(1, n, n), dtype=np.dtype("float32"), chunks=(1, n, n)
                ),
                attrs={
                    "long_name": "phis",
                    "units": "none",
                    "checksum": "86624B29A22B1115",
                },
            ),
        },
    )


def generate_restart_directory(output: str, resolution: str = "C48"):
    pass


def _read_metadata_remote(fs, url):
    with fs.open(url, "rb") as f:
        return xr.open_dataset(f)


def remote_nc_to_schema(url: str) -> synth.DatasetSchema:
    fs = vcm.cloud.get_fs(url)
    meta = _read_metadata_remote(fs, url)
    return synth.read_schema_from_dataset(meta)
