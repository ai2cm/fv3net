import vcm
import synth
from synth import DatasetSchema, CoordinateSchema, ChunkedArray, VariableSchema
import numpy as np
import xarray as xr


def fv_core_schema(n: int, nz: int, x, xi, y, yi, z):
    def CENTERED(name: str):
        return VariableSchema(
            name=name,
            dims=["Time", z, y, x],
            array=ChunkedArray(
                shape=(1, nz, n, n), dtype=np.dtype("float32"), chunks=(1, nz, n, n),
            ),
            attrs={"long_name": name, "units": "none",},
        )

    def Y_OUTER(name: str):
        return VariableSchema(
            name=name,
            dims=["Time", z, y, xi],
            array=ChunkedArray(
                shape=(1, nz, n, n + 1),
                dtype=np.dtype("float32"),
                chunks=(1, nz, n, n + 1),
            ),
            attrs={"long_name": name, "units": "none",},
        )

    def X_OUTER(name: str):
        return VariableSchema(
            name=name,
            dims=["Time", z, yi, x],
            array=ChunkedArray(
                shape=(1, nz, n + 1, n),
                dtype=np.dtype("float32"),
                chunks=(1, nz, n + 1, n),
            ),
            attrs={"long_name": name, "units": "none",},
        )

    def SURFACE(name: str):
        return VariableSchema(
            name=name,
            dims=["Time", y, x],
            array=ChunkedArray(
                shape=(1, n, n), dtype=np.dtype("float32"), chunks=(1, n, n)
            ),
            attrs={"long_name": name, "units": "none",},
        )

    variables_spec = [
        ("u", X_OUTER),
        ("v", Y_OUTER),
        ("W", CENTERED),
        ("DZ", CENTERED),
        ("T", CENTERED),
        ("delp", CENTERED),
        ("phis", SURFACE),
    ]

    variables = {}
    for name, func in variables_spec:
        variables[name] = func(name)

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
        variables=variables,
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
