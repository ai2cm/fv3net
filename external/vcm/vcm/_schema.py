import numpy as np
from typing import Sequence, Any, Mapping
import xarray as xr
from dataclasses import dataclass
from vcm.cubedsphere.constants import (
    COORD_X_CENTER,
    COORD_X_OUTER,
    COORD_Y_CENTER,
    COORD_Y_OUTER,
)

print("SCHEMA!!!")


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

    @staticmethod
    def from_dataarray(data: xr.DataArray) -> "Schema":
        return Schema(data.dims, data.dtype)

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

    def rename(self, arr: xr.DataArray) -> xr.DataArray:
        """Apply metadata to array_like object

        Args:
            arr (array_like): array to apply the dimension names to. If the number of
                dimensions

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


def dataset_to_schema(ds: xr.Dataset) -> Mapping[Any, Schema]:
    return {variable: Schema.from_dataarray(ds[variable]) for variable in ds}


def rename_dataset(ds: xr.Dataset):
    return xr.Dataset(
        {
            variable: REGISTRY[variable].rename(ds[variable])
            if variable in REGISTRY
            else ds[variable]
            for variable in ds
        }
    )


# List of schema
REGISTRY = {
    "phis": Schema(dims=(..., "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")),
    "delp": Schema(
        dims=(..., "pfull", "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")
    ),
    "DZ": Schema(dims=(..., "pfull", "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")),
    "W": Schema(dims=(..., "pfull", "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")),
    "T": Schema(dims=(..., "pfull", "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")),
    "u": Schema(dims=(..., "pfull", "COORD_Y_OUTER", "COORD_X_CENTER"), dtype=np.dtype("float32")),
    "v": Schema(dims=(..., "pfull", "COORD_Y_CENTER", "COORD_X_OUTER"), dtype=np.dtype("float32")),
    "u_srf": Schema(dims=(..., "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")),
    "v_srf": Schema(dims=(..., "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")),
    "cld_amt": Schema(
        dims=(..., "pfull", "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")
    ),
    "sphum": Schema(
        dims=(..., "pfull", "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")
    ),
    "liq_wat": Schema(
        dims=(..., "pfull", "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")
    ),
    "rainwat": Schema(
        dims=(..., "pfull", "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")
    ),
    "ice_wat": Schema(
        dims=(..., "pfull", "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")
    ),
    "snowwat": Schema(
        dims=(..., "pfull", "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")
    ),
    "graupel": Schema(
        dims=(..., "pfull", "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")
    ),
    "o3mr": Schema(
        dims=(..., "pfull", "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")
    ),
    "sgs_tke": Schema(
        dims=(..., "pfull", "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")
    ),
    "slmsk": Schema(dims=(..., "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")),
    "tsea": Schema(dims=(..., "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")),
    "sheleg": Schema(dims=(..., "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")),
    "tg3": Schema(dims=(..., "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")),
    "zorl": Schema(dims=(..., "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")),
    "alvsf": Schema(dims=(..., "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")),
    "alvwf": Schema(dims=(..., "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")),
    "alnsf": Schema(dims=(..., "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")),
    "alnwf": Schema(dims=(..., "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")),
    "facsf": Schema(dims=(..., "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")),
    "facwf": Schema(dims=(..., "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")),
    "vfrac": Schema(dims=(..., "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")),
    "canopy": Schema(dims=(..., "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")),
    "f10m": Schema(dims=(..., "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")),
    "t2m": Schema(dims=(..., "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")),
    "q2m": Schema(dims=(..., "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")),
    "vtype": Schema(dims=(..., "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")),
    "stype": Schema(dims=(..., "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")),
    "uustar": Schema(dims=(..., "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")),
    "ffmm": Schema(dims=(..., "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")),
    "ffhh": Schema(dims=(..., "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")),
    "hice": Schema(dims=(..., "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")),
    "fice": Schema(dims=(..., "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")),
    "tisfc": Schema(dims=(..., "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")),
    "tprcp": Schema(dims=(..., "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")),
    "srflag": Schema(dims=(..., "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")),
    "snwdph": Schema(dims=(..., "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")),
    "shdmin": Schema(dims=(..., "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")),
    "shdmax": Schema(dims=(..., "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")),
    "slope": Schema(dims=(..., "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")),
    "snoalb": Schema(dims=(..., "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")),
    "sncovr": Schema(dims=(..., "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")),
    "stc": Schema(dims=(..., "phalf", "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")),
    "smc": Schema(dims=(..., "phalf", "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")),
    "slc": Schema(dims=(..., "phalf", "COORD_Y_CENTER", "COORD_X_CENTER"), dtype=np.dtype("float32")),
}
