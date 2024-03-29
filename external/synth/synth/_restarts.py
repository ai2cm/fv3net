import synth
from typing import Iterable, Mapping, Sequence
from synth.core import DatasetSchema, CoordinateSchema, ChunkedArray, VariableSchema
import numpy as np
import xarray as xr

from toolz import valmap


def generate_restart_data(
    nx: int = 48, nz: int = 79, n_soil: int = 4, include_agrid_winds: bool = False
) -> Mapping[str, Mapping[int, xr.Dataset]]:
    """Generate a set of fake restart data for testing purposes

    Args:
        n: the extent of tile. For instance for C48, n=48
        nz: the number of height levels
        n_soil: the number of soil levels
        include_agrid_winds: whether to include "ua" and "va" in fv_core.res files

    Returns:
        restarts: collection of restart data as a doubly-nested mapping, whose first
             key is the filename without extension (one of "fv_core.res",
             "fv_tracer.res", "fv_srf_wnd.res", or "sfc_data"), second key is
             tile number, and value is the xr.Dataset which would be
             contained in the file these keys indicate. Example::

                category = 'fv_core.res'
                tile = 4
                # Fourth tile of fv_core.res data:
                restarts[category][tile]

    """
    tiles = range(1, 7)
    ranges = {
        # Need to use a small range here to avoid SEGFAULTS in the mappm
        # if delp varies to much then the mean pressures may lie completely out
        # of bounds an individual column
        "delp": synth.Range(0.99, 1.01)
    }

    schema = {
        "fv_core.res": _fv_core_schema(nx, nz, include_agrid_winds),
        "sfc_data": _sfc_data(nx, n_soil),
        "fv_tracer.res": _fv_tracer_schema(nx, nz),
        "fv_srf_wnd.res": _fv_srf_wnd_schema(nx),
    }

    def _generate_from_schema(schema: DatasetSchema):
        return {tile: synth.generate(schema, ranges) for tile in tiles}

    return valmap(_generate_from_schema, schema)


def _range(n):
    return np.arange(1, n + 1).astype("f4")


class _RestartCategorySchemaFactory:
    def __init__(
        self,
        x: str = None,
        xi: str = None,
        y: str = None,
        yi: str = None,
        z: str = None,
        n: int = None,
        nz: int = None,
    ):
        self.x = x
        self.y = y
        self.xi = xi
        self.yi = yi
        self.z = z
        self.n = n
        self.nz = nz

    @property
    def x_coord(self) -> CoordinateSchema:
        x = self.x
        n = self.n
        return CoordinateSchema(
            name=x,
            dims=[x],
            value=_range(n),
            attrs={"long_name": x, "units": "none", "cartesian_axis": "X"},
        )

    @property
    def xi_coord(self) -> CoordinateSchema:
        n = self.n
        xi = self.xi
        return CoordinateSchema(
            name=xi,
            dims=[xi],
            value=_range(n + 1),
            attrs={"long_name": xi, "units": "none", "cartesian_axis": "X"},
        )

    @property
    def yi_coord(self) -> CoordinateSchema:
        yi = self.yi
        n = self.n
        return CoordinateSchema(
            name=yi,
            dims=[yi],
            value=_range(n + 1),
            attrs={"long_name": yi, "units": "none", "cartesian_axis": "Y"},
        )

    @property
    def y_coord(self) -> CoordinateSchema:

        y = self.y
        n = self.n
        return CoordinateSchema(
            name=y,
            dims=[y],
            value=_range(n),
            attrs={"long_name": y, "units": "none", "cartesian_axis": "Y"},
        )

    @property
    def z_coord(self) -> CoordinateSchema:
        z = self.z
        nz = self.nz
        return CoordinateSchema(
            name=z,
            dims=[z],
            value=_range(nz),
            attrs={"long_name": z, "units": "none", "cartesian_axis": "Z"},
        )

    @property
    def time_coord(self) -> CoordinateSchema:
        return CoordinateSchema(
            name="Time",
            dims=["Time"],
            value=np.array([1.0], dtype="f4"),
            attrs={"long_name": "Time", "units": "time level", "cartesian_axis": "T"},
        )

    def centered(self, name: str) -> VariableSchema:
        return VariableSchema(
            name=name,
            dims=["Time", self.z, self.y, self.x],
            array=ChunkedArray(
                shape=(1, self.nz, self.n, self.n),
                dtype=np.dtype("float32"),
                chunks=(1, self.nz, self.n, self.n),
            ),
            attrs={"long_name": name, "units": "none"},
        )

    def x_outer(self, name: str) -> VariableSchema:
        return VariableSchema(
            name=name,
            dims=["Time", self.z, self.y, self.xi],
            array=ChunkedArray(
                shape=(1, self.nz, self.n, self.n + 1),
                dtype=np.dtype("float32"),
                chunks=(1, self.nz, self.n, self.n + 1),
            ),
            attrs={"long_name": name, "units": "none"},
        )

    def y_outer(self, name: str) -> VariableSchema:
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
            attrs={"long_name": name, "units": "none"},
        )

    def surface(self, name: str) -> VariableSchema:
        n = self.n
        y = self.y
        x = self.x

        return VariableSchema(
            name=name,
            dims=["Time", y, x],
            array=ChunkedArray(
                shape=(1, n, n), dtype=np.dtype("float32"), chunks=(1, n, n)
            ),
            attrs={"long_name": name, "units": "none"},
        )

    def _generate_variables(
        self,
        centered: Iterable[str],
        y_outer: Iterable[str],
        x_outer: Iterable[str],
        surface: Iterable[str],
    ) -> Mapping[str, VariableSchema]:
        output = {}

        for variable_list, schema_func in (
            (centered, self.centered),
            (x_outer, self.x_outer),
            (y_outer, self.y_outer),
            (surface, self.surface),
        ):
            for variable in variable_list:
                output[variable] = schema_func(variable)

        return output

    def _generate_coords(
        self,
        centered: Sequence[str],
        y_outer: Sequence[str],
        x_outer: Sequence[str],
        surface: Sequence[str],
    ) -> Mapping[str, CoordinateSchema]:
        output = {}
        output["Time"] = self.time_coord
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
        centered: Sequence[str] = (),
        y_outer: Sequence[str] = (),
        x_outer: Sequence[str] = (),
        surface: Sequence[str] = (),
    ) -> DatasetSchema:
        coords = self._generate_coords(centered, y_outer, x_outer, surface)
        variables = self._generate_variables(centered, y_outer, x_outer, surface)

        return DatasetSchema(variables=variables, coords=coords)


def _fv_core_schema(
    n: int, nz: int, include_agrid_winds: bool = False
) -> DatasetSchema:
    if include_agrid_winds:
        centered = ["W", "DZ", "T", "delp", "ua", "va"]
    else:
        centered = ["W", "DZ", "T", "delp"]
    return _RestartCategorySchemaFactory(
        n=n, nz=nz, x="xaxis_1", xi="xaxis_2", y="yaxis_2", yi="yaxis_1", z="zaxis_1"
    ).generate(centered=centered, y_outer=["u"], x_outer=["v"], surface=["phis"],)


def _fv_tracer_schema(n: int, nz: int) -> DatasetSchema:
    return _RestartCategorySchemaFactory(
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


def _fv_srf_wnd_schema(n: int) -> DatasetSchema:
    return _RestartCategorySchemaFactory(n=n, x="xaxis_1", y="yaxis_1").generate(
        surface=["u_srf", "v_srf"]
    )


def _sfc_data(n: int, n_soil: int) -> DatasetSchema:
    return _RestartCategorySchemaFactory(
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
        centered=["stc", "smc", "slc"],
    )
