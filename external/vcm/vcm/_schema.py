import numpy as np
from typing import Sequence, Any, Mapping, Dict
import xarray as xr
from dataclasses import dataclass
from vcm.cubedsphere.constants import (
    COORD_X_CENTER,
    COORD_X_OUTER,
    COORD_Y_CENTER,
    COORD_Y_OUTER,
    COORD_Z_CENTER,
    COORD_Z_SOIL,
)


@dataclass
class Schema:
    """An object for representing the metadata, dimensions, and dtype of a DataArray

    Attributes:
        dims: a list of dimension names. If this begins with ``...``, then
            any data array ending with the dimensions after "..." will validate
            against this schema.

    """

    dims: Sequence[Any]
    dtype: np.dtype
    attrs: Dict

    @staticmethod
    def from_dataarray(data: xr.DataArray) -> "Schema":
        return Schema(data.dims, data.dtype, data.attrs)

    def validate(self, data: xr.DataArray) -> bool:
        return self._validate_dims(data) and self._validate_dtype(data)

    def _validate_dtype(self, data: xr.DataArray) -> bool:
        return self.dtype == data.dtype

    def _validate_dims(self, data: xr.DataArray) -> bool:
        if self.dims[0] is ...:
            expected_dims = self.dims[1:]
            n = len(expected_dims)
            data_dims = data.dims[-n:]
        else:
            expected_dims = self.dims
            data_dims = data.dims
        return set(expected_dims) == set(data_dims)

    def coerce_dataarray_to_schema(self, arr: xr.DataArray) -> xr.DataArray:
        """Apply dimension names and variable attributes to array_like object

        Args:
            arr (array_like): array to which to apply the metadata.

        Returns:
            arr: (array_like): An array with renamed dimensions and attributes.

        """

        arr = self._rename_dimensions(arr)
        return self._add_long_name_and_units(arr)

    def _rename_dimensions(self, arr: xr.DataArray) -> xr.DataArray:
        """Apply dimension names to array_like object

        Args:
            arr (array_like): array to apply the dimension names to. If the number of
                dimensions doesn't match an error is returned. If the schema
                specifies only n trailing dimensions then only n trailing
                dimensions are renamed.

        Returns:
            arr: (array_like): An array with renamed dimensions.

        """

        if self.dims[0] is ...:
            dims = self.dims[1:]
            dims = tuple(arr.dims[: -len(dims)]) + tuple(dims)
        else:
            dims = self.dims

        if len(dims) != len(arr.dims):
            raise ValueError(
                "schema dimensions must have the same length as "
                "arr or begin with ``...``"
            )

        rename_dict = dict(zip(arr.dims, dims))
        return arr.rename(rename_dict)

    def _add_long_name_and_units(self, arr: xr.DataArray) -> xr.DataArray:
        """Apply attributes to array_like object

        Args:
            arr (array_like): array to whicn to apply the schema's attributes.

        Returns:
            arr: (array_like): An array with attributes set.

        """

        if self.attrs["long_name"] is not None:
            arr.attrs["long_name"] = self.attrs["long_name"]
        if self.attrs["units"] is not None:
            arr.attrs["units"] = self.attrs["units"]

        return arr


def dataset_to_schema(ds: xr.Dataset) -> Mapping[Any, Schema]:
    return {variable: Schema.from_dataarray(ds[variable]) for variable in ds}


def coerce_dataset_to_schema(ds: xr.Dataset):
    return xr.Dataset(
        {
            variable: REGISTRY[variable].coerce_dataarray_to_schema(ds[variable])
            if variable in REGISTRY
            else ds[variable]
            for variable in ds
        }
    )


# List of schema
# Note that long names and units are taken from various sources and may be
# incomplete/incorrect, so use with caution

REGISTRY = {
    "phis": Schema(
        dims=(..., COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={"long_name": "surface geopotential", "units": "m ** 2 / s ** 2"},
    ),
    "delp": Schema(
        dims=(..., COORD_Z_CENTER, COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={"long_name": "pressure thickness", "units": "Pa"},
    ),
    "DZ": Schema(
        dims=(..., COORD_Z_CENTER, COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={"long_name": "vertical thickness", "units": "m"},
    ),
    "T": Schema(
        dims=(..., COORD_Z_CENTER, COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={"long_name": "temperature", "units": "K"},
    ),
    "W": Schema(
        dims=(..., COORD_Z_CENTER, COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={"long_name": "vertical wind", "units": "m / s"},
    ),
    "u": Schema(
        dims=(..., COORD_Z_CENTER, COORD_Y_OUTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={"long_name": "zonal wind on the D grid", "units": "m / s"},
    ),
    "v": Schema(
        dims=(..., COORD_Z_CENTER, COORD_Y_CENTER, COORD_X_OUTER),
        dtype=np.dtype("float32"),
        attrs={"long_name": "meridional wind on the D grid", "units": "m / s"},
    ),
    "u_srf": Schema(
        dims=(..., COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={"long_name": "zonal surface wind", "units": "m / s"},
    ),
    "v_srf": Schema(
        dims=(..., COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={"long_name": "meridional surface wind", "units": "m / s"},
    ),
    "cld_amt": Schema(
        dims=(..., COORD_Z_CENTER, COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={"long_name": "cloud fraction", "units": "fraction"},
    ),
    "sphum": Schema(
        dims=(..., COORD_Z_CENTER, COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={"long_name": "specific humidity", "units": "kg / kg"},
    ),
    "liq_wat": Schema(
        dims=(..., COORD_Z_CENTER, COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={"long_name": "liquid water", "units": "kg / kg"},
    ),
    "rainwat": Schema(
        dims=(..., COORD_Z_CENTER, COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={"long_name": "rain water", "units": "kg / kg"},
    ),
    "ice_wat": Schema(
        dims=(..., COORD_Z_CENTER, COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={"long_name": "ice water", "units": "kg / kg"},
    ),
    "snowwat": Schema(
        dims=(..., COORD_Z_CENTER, COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={"long_name": "snow water", "units": "kg / kg"},
    ),
    "graupel": Schema(
        dims=(..., COORD_Z_CENTER, COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={"long_name": "graupel water", "units": "kg / kg"},
    ),
    "o3mr": Schema(
        dims=(..., COORD_Z_CENTER, COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={"long_name": "ozone", "units": "kg / kg"},
    ),
    "sgs_tke": Schema(
        dims=(..., COORD_Z_CENTER, COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={
            "long_name": "subgrid-scale turbulent kinetic energy",
            "units": "m ** 2 / s ** 2",
        },
    ),
    "slmsk": Schema(
        dims=(..., COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={
            "long_name": "sea/land/sea ice mask array",
            "units": "sea:0, land:1, sea-ice:2",
        },
    ),
    "tsea": Schema(
        dims=(..., COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={"long_name": "sea surface temperature", "units": "K"},
    ),
    "sheleg": Schema(
        dims=(..., COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={"long_name": "snow water equivalent", "units": "mm"},
    ),
    "tg3": Schema(
        dims=(..., COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={"long_name": "deep soil temperature", "units": "K"},
    ),
    "zorl": Schema(
        dims=(..., COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={"long_name": "surface roughness", "units": "cm"},
    ),
    "alvsf": Schema(
        dims=(..., COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={
            "long_name": "mean vis albedo with strong cosz dependency",
            "units": "dimensionless",
        },
    ),
    "alvwf": Schema(
        dims=(..., COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={
            "long_name": "mean vis albedo with weak cosz dependency",
            "units": "dimensionless",
        },
    ),
    "alnsf": Schema(
        dims=(..., COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={
            "long_name": "mean nir albedo with strong cosz dependency",
            "units": "dimensionless",
        },
    ),
    "alnwf": Schema(
        dims=(..., COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={
            "long_name": "mean nir albedo with weak cosz dependency",
            "units": "dimensionless",
        },
    ),
    "facsf": Schema(
        dims=(..., COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={
            "long_name": "fractional coverage with strong cosz dependency",
            "units": "dimensionless",
        },
    ),
    "facwf": Schema(
        dims=(..., COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={
            "long_name": "fractional coverage with weak cosz dependency",
            "units": "dimensionless",
        },
    ),
    "vfrac": Schema(
        dims=(..., COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={"long_name": "vegetation fraction", "units": "dimensionless"},
    ),
    "canopy": Schema(
        dims=(..., COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={"long_name": "canopy water", "units": "cm?"},
    ),
    "f10m": Schema(
        dims=(..., COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={
            "long_name": "fm at 10m - Ratio of sigma level 1 wind and 10m wind",
            "units": "dimensionless",
        },
    ),
    "q2m": Schema(
        dims=(..., COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={"long_name": "2 meter humidity", "units": "kg / kg"},
    ),
    "t2m": Schema(
        dims=(..., COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={"long_name": "2 meter temperature", "units": "K"},
    ),
    "vtype": Schema(
        dims=(..., COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={"long_name": "vegetation type", "units": "categorical"},
    ),
    "stype": Schema(
        dims=(..., COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={"long_name": "soil type", "units": "categorical"},
    ),
    "uustar": Schema(
        dims=(..., COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={"long_name": "friction velocity squared", "units": "'m ** 2 / s ** 2'}"},
    ),
    "ffmm": Schema(
        dims=(..., COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={"long_name": "fm parameter from PBL scheme", "units": "?"},
    ),
    "ffhh": Schema(
        dims=(..., COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={"long_name": "fh parameter from PBL scheme", "units": "?"},
    ),
    "hice": Schema(
        dims=(..., COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={"long_name": "sea ice thickness", "units": "m"},
    ),
    "fice": Schema(
        dims=(..., COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={
            "long_name": "ice fraction over open water grid",
            "units": "dimensionless",
        },
    ),
    "tisfc": Schema(
        dims=(..., COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={"long_name": "surface temperature over ice fraction", "units": "K"},
    ),
    "tprcp": Schema(
        dims=(..., COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={"long_name": "total precipitations in timestep", "units": "kg / m**2"},
    ),
    "srflag": Schema(
        dims=(..., COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={
            "long_name": "snow/rain flag for precipitation",
            "units": "1: snow, 0: rain",
        },
    ),
    "snwdph": Schema(
        dims=(..., COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={"long_name": "snow depth", "units": "mm"},
    ),
    "shdmin": Schema(
        dims=(..., COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={
            "long_name": "min fractional coverage of green vegetation",
            "units": "dimensionless",
        },
    ),
    "shdmax": Schema(
        dims=(..., COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={
            "long_name": "max fractional coverage of green vegetation",
            "units": "dimensionless",
        },
    ),
    "slope": Schema(
        dims=(..., COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={"long_name": "surface slope type for lsm", "units": "categorical"},
    ),
    "snoalb": Schema(
        dims=(..., COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={
            "long_name": "maximum snow albedo in fraction",
            "units": "dimensionless",
        },
    ),
    "sncovr": Schema(
        dims=(..., COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={"long_name": "snow cover in fraction", "units": "dimensionless"},
    ),
    "stc": Schema(
        dims=(..., COORD_Z_SOIL, COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={"long_name": "soil temperature", "units": "K"},
    ),
    "smc": Schema(
        dims=(..., COORD_Z_SOIL, COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={"long_name": "total soil moisture", "units": "volumetric water content"},
    ),
    "slc": Schema(
        dims=(..., COORD_Z_SOIL, COORD_Y_CENTER, COORD_X_CENTER),
        dtype=np.dtype("float32"),
        attrs={
            "long_name": "liquid soil moisture",
            "units": "volumetric water content",
        },
    ),
}
