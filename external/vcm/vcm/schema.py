from typing import Sequence, Any, Mapping, Dict
import xarray as xr
from dataclasses import dataclass


@dataclass
class Schema:
    """An object for representing the metadata, dimensions, and dtype of a DataArray

    Attributes:
        dims: a list of dimension names. If this begins with ``...``, then
            any data array ending with the dimensions after "..." will validate
            against this schema.

    """

    dims: Sequence[Any]
    attrs: Dict

    @staticmethod
    def from_dataarray(data: xr.DataArray) -> "Schema":
        return Schema(data.dims, data.attrs)

    def validate(self, data: xr.DataArray) -> bool:
        return self._validate_dims(data)

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