import vcm
import synth
from typing import Iterable, Mapping, List
from synth import DatasetSchema, CoordinateSchema, ChunkedArray, VariableSchema
import numpy as np
import xarray as xr

from toolz import valmap

from collections import namedtuple


def _range(n) -> List[float]:
    return np.arange(1, n + 1).astype(float).tolist()


class RestartCategorySchemaFactory:
    def __init__(self, x=None, xi=None, y=None, yi=None, z=None, n=None, nz=None):
        self.x = x
        self.y = y
        self.xi = xi
        self.yi = yi
        self.z = z
        self.n = n
        self.nz = nz

    @property
    def x_coord(self):
        x = self.x
        n = self.n
        return (
            CoordinateSchema(
                name=x,
                dims=[x],
                value=_range(n),
                attrs={"long_name": x, "units": "none", "cartesian_axis": "X"},
            ),
        )

    @property
    def xi_coord(self):
        n = self.n
        xi = self.xi
        return CoordinateSchema(
            name=xi,
            dims=[xi],
            value=_range(n + 1),
            attrs={"long_name": xi, "units": "none", "cartesian_axis": "X"},
        )

    @property
    def yi_coord(self):
        yi = self.yi
        n = self.n
        return CoordinateSchema(
            name=yi,
            dims=[yi],
            value=_range(n + 1),
            attrs={"long_name": yi, "units": "none", "cartesian_axis": "Y"},
        )

    @property
    def y_coord(self):

        y = self.y
        n = self.n
        return (
            CoordinateSchema(
                name=y,
                dims=[y],
                value=_range(n),
                attrs={"long_name": y, "units": "none", "cartesian_axis": "Y"},
            ),
        )

    @property
    def z_coord(self):
        z = self.z
        nz = self.nz
        return (
            CoordinateSchema(
                name=z,
                dims=[z],
                value=_range(nz),
                attrs={"long_name": z, "units": "none", "cartesian_axis": "Z"},
            ),
        )

    @property
    def time_coord(self):
        return (
            CoordinateSchema(
                name="Time",
                dims=["Time"],
                value=[1.0],
                attrs={
                    "long_name": "Time",
                    "units": "time level",
                    "cartesian_axis": "T",
                },
            ),
        )

    def centered(self, name: str):
        return VariableSchema(
            name=name,
            dims=["Time", self.z, self.y, self.x],
            array=ChunkedArray(
                shape=(1, self.nz, self.n, self.n),
                dtype=np.dtype("float32"),
                chunks=(1, self.nz, self.n, self.n),
            ),
            attrs={"long_name": name, "units": "none",},
        )

    def x_outer(self, name: str):
        return VariableSchema(
            name=name,
            dims=["Time", self.z, self.y, self.xi],
            array=ChunkedArray(
                shape=(1, self.nz, self.n, self.n + 1),
                dtype=np.dtype("float32"),
                chunks=(1, self.nz, self.n, self.n + 1),
            ),
            attrs={"long_name": name, "units": "none",},
        )

    def y_outer(self, name: str):
        n = self.n
        nz = self.nz
        z = self.z
        yi = self.yi
        x = self.x
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

    def surface(self, name: str):
        n = self.n
        y = self.y
        x = self.x

        return VariableSchema(
            name=name,
            dims=["Time", y, x],
            array=ChunkedArray(
                shape=(1, n, n), dtype=np.dtype("float32"), chunks=(1, n, n)
            ),
            attrs={"long_name": name, "units": "none",},
        )

    def _generate_variables(
        self,
        centered: Iterable[str],
        y_outer: Iterable[str],
        x_outer: Iterable[str],
        surface: Iterable[str],
    ) -> Mapping[str, VariableSchema]:
        output = {}

        for variable in centered:
            output[variable] = self.centered(variable)

        for variable in x_outer:
            output[variable] = self.x_outer(variable)

        for variable in y_outer:
            output[variable] = self.y_outer(variable)

        for variable in surface:
            output[variable] = self.surface(variable)

        return output

    def _generate_coords(
        self,
        centered: Iterable[str],
        y_outer: Iterable[str],
        x_outer: Iterable[str],
        surface: Iterable[str],
    ) -> Mapping[str, CoordinateSchema]:
        output = {}
        if len(centered) > 0:
            output[self.x] = self.x_coord
            output[self.y] = self.y_coord
            output[self.z] = self.z_coord

        if len(x_outer) > 0:
            output[self.xi] = self.xi_coord
            output[self.y] = self.y_coord
            output[self.z] = self.z_coord

        if len(y_outer) > 0:
            output[self.x] = self.x_coord
            output[self.yi] = self.yi_coord
            output[self.z] = self.z_coord

        if len(surface) > 0:
            output[self.x] = self.x_coord
            output[self.y] = self.y_coord

        return output

    def generate(
        self,
        centered: Iterable[str] = (),
        y_outer: Iterable[str] = (),
        x_outer: Iterable[str] = (),
        surface: Iterable[str] = (),
    ) -> DatasetSchema:
        coords = self._generate_coords(centered, x_outer, y_outer, surface)
        variables = self._generate_variables(centered, x_outer, y_outer, surface)

        return DatasetSchema(variables=variables, coords=coords)


def fv_core_schema(n: int, nz: int):
    return RestartCategorySchemaFactory(
        n=n, nz=nz, x="xaxis_1", xi="xaxis_2", y="yaxis_2", yi="yaxis_1", z="zaxis_1"
    ).generate(
        centered=["W", "DZ", "T", "delp"],
        y_outer=["u"],
        x_outer=["v"],
        surface=["delp"],
    )


def fv_tracer_schema(n: int, nz: int):
    return RestartCategorySchemaFactory(
        n=n, nz=nz, x="xaxis_1", y="yaxis_1", z="zaxis_1"
    ).generate(
        centered=[
            "sphum",
            "liq_wat",
            "rainwat",
            "ice_wat",
            "snowwat",
            "graupel",
            "o3mr",
            "sgs_tke",
            "cld_amt",
        ],
    )


def fv_srf_wnd_schema(n: int):
    return RestartCategorySchemaFactory(n=n, x="xaxis_1", y="yaxis_1").generate(
        surface=["u_srf", "v_srf"]
    )


def sfc_data(n: int, n_soil: int):
    return RestartCategorySchemaFactory(
        n=n, nz=n_soil, x="xaxis_1", y="yaxis_1", z="zaxis_1"
    ).generate(
        surface=[
            "slmsk",
            "tsea",
            "sheleg",
            "tg3",
            "zorl",
            "alvsf",
            "alvwf",
            "alnsf",
            "alnwf",
            "facsf",
            "facwf",
            "vfrac",
            "canopy",
            "f10m",
            "t2m",
            "q2m",
            "vtype",
            "stype",
            "uustar",
            "ffmm",
            "ffhh",
            "hice",
            "fice",
            "tisfc",
            "tprcp",
            "srflag",
            "snwdph",
            "shdmin",
            "shdmax",
            "slope",
            "snoalb",
            "sncovr",
        ],
        centered=["stc", "smc", "slc",],
    )


def generate_restart_data(
    n=48, nz=79, n_soil=4
) -> Mapping[str, Mapping[int, xr.Dataset]]:
    tiles = [1, 2, 3, 4, 5, 6]

    def _generate_from_schema(schema: DatasetSchema):
        return {tile: synth.generate(schema) for tile in tiles}

    schema = {
        "fv_core.res": fv_core_schema(n, nz),
        "sfc_data": sfc_data(n, n_soil),
        "fv_tracer.res": fv_tracer_schema(n, nz),
        "fv_src_wnd.res": fv_srf_wnd_schema(n),
    }

    return valmap(_generate_from_schema, schema)


def _read_metadata_remote(fs, url):
    with fs.open(url, "rb") as f:
        return xr.open_dataset(f)


def remote_nc_to_schema(url: str) -> synth.DatasetSchema:
    fs = vcm.cloud.get_fs(url)
    meta = _read_metadata_remote(fs, url)
    return synth.read_schema_from_dataset(meta)
