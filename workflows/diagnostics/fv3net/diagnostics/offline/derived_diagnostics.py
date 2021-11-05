from typing import Sequence, Tuple

import xarray as xr

import vcm

from fv3net.diagnostics._shared.registry import Registry
from fv3net.diagnostics._shared.constants import WVP, COL_DRYING


def merge_derived(diags: Sequence[Tuple[str, xr.DataArray]]) -> xr.Dataset:
    out = xr.Dataset()
    for name, da in diags:
        if len(da.dims) != 0:
            # don't add empty DataArrays, which are returned in case of missing inputs
            out[name] = da
    return out


# all functions added to this registry must take a single xarray Dataset as
# input and return a single xarray DataArray
derived_registry = Registry(merge_derived)


@derived_registry.register(f"conditional_average_of_{COL_DRYING}_on_{WVP}")
def conditional_average(diags: xr.Dataset) -> xr.DataArray:
    count_name = f"{WVP}_versus_{COL_DRYING}_hist_2d"
    q2_bin_name = f"{COL_DRYING}_bins"
    q2_bin_widths_name = f"{COL_DRYING}_bin_width_hist_2d"
    if count_name not in diags:
        return xr.DataArray()
    count = diags[count_name]
    q2_bin_centers = diags[q2_bin_name] + 0.5 * diags[q2_bin_widths_name]
    average = vcm.weighted_average(q2_bin_centers, count, dims=[q2_bin_name])
    return average.assign_attrs(
        long_name="Conditional average of -<Q2> on water vapor path", units="mm/day"
    )
