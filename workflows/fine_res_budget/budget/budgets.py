from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import xarray as xr

from vcm import pressure_at_interface
from vcm.cubedsphere import block_upsample, weighted_block_average, regrid_vertical

import logging

logger = logging.getLogger(__name__)


@dataclass
class Grid:
    """Convenience interface for doing grid-math
    
    This class supplies the dimension name arguments to the cubedsphere methods,
    which are the same for all computations done here.
    
    """

    x: str
    y: str
    z: str
    xi: str
    yi: str
    zi: str

    def pressure_at_interface(self, p):
        return pressure_at_interface(p, dim_center=self.z, dim_outer=self.zi)

    def weighted_block_average(self, delp, weight, factor):
        return weighted_block_average(delp, weight, factor, x_dim=self.x, y_dim=self.y)

    def block_upsample(self, coarse, factor):
        return block_upsample(coarse, factor, [self.x, self.y])

    def regrid_vertical(self, *args, **kwargs):
        """regrid_vertical(
            p_in: xarray.core.dataarray.DataArray,
            f_in: xarray.core.dataarray.DataArray,
            p_out: xarray.core.dataarray.DataArray,
        """
        return regrid_vertical(
            *args, z_dim_center=self.z, z_dim_outer=self.zi, **kwargs
        )

    def pressure_level_average(self, delp_fine, delp_coarse, area, field, factor):
        pi = self.pressure_at_interface(delp_fine)
        pi_c = self.pressure_at_interface(delp_coarse)
        pi_c_up = self.block_upsample(pi_c, factor=factor)

        fg = self.regrid_vertical(pi, field, pi_c_up)
        avg = self.weighted_block_average(fg, area, factor)
        return avg.drop_vars([self.x, self.y, self.z], errors="ignore").rename(
            field.name
        )


def storage(field: xr.DataArray, time_step: float) -> xr.DataArray:
    return (field.sel(step="end") - field.sel(step="begin")) / time_step


GRID = Grid("grid_xt", "grid_yt", "pfull", "grid_x", "grid_y", "pfulli")


def coarsen_variables(
    fields: Iterable[xr.DataArray],
    delp_fine: xr.DataArray,
    delp_coarse: xr.DataArray,
    area: xr.DataArray,
    factor: int,
):
    """Coarsen an iterable of DataArrays on surfaces of constant pressure.

    Args:
        fields: iterable of DataArrays.
        delp_fine: DataArray containing delp on the fine grid.
        delp_coarse: DataArray containing delp on the coarse grid.
        area: DataArray containing surface area on the fine grid.
        factor: Integer coarsening factor.

    Returns:
        xr.Dataset containing the coarsened variables.
    """
    return xr.merge(
        [
            GRID.pressure_level_average(delp_fine, delp_coarse, area, field, factor)
            for field in fields
        ]
    )


def _infer_second_moment_name(field_1: xr.DataArray, field_2: xr.DataArray) -> str:
    """Infer the units for the product of two DataArrays."""
    return f"{field_1.name}_{field_2.name}"


def _infer_second_moment_units(field_1: xr.DataArray, field_2: xr.DataArray) -> str:
    """Infer the units for the product of two DataArrays.  Just does the naive
    thing of appending one set of units after the other."""
    field_1_units = field_1.attrs.get("units", "")
    field_2_units = field_2.attrs.get("units", "")
    return f"{field_1_units} {field_2_units}".strip()


def _infer_second_moment_long_name(field_1: xr.DataArray, field_2: xr.DataArray) -> str:
    """Infer the longname for the product of two DataArrays."""
    field_1_long_name = field_1.attrs.get("long_name", field_1.name)
    field_2_long_name = field_2.attrs.get("long_name", field_2.name)
    return f"Product of {field_1_long_name} and {field_2_long_name}"


def _infer_second_moment_attrs(
    field_1: xr.DataArray, field_2: xr.DataArray
) -> Dict[str, str]:
    """Infer the attributes for the product of two DataArrays."""
    units = _infer_second_moment_units(field_1, field_2)
    long_name = _infer_second_moment_long_name(field_1, field_2)
    return dict(units=units, long_name=long_name)


def _compute_second_moment(
    field_1: xr.DataArray, field_2: xr.DataArray
) -> xr.DataArray:
    """Compute the product of two DataArrays, adding appropriate metadata."""
    name = _infer_second_moment_name(field_1, field_2)
    attrs = _infer_second_moment_attrs(field_1, field_2)
    return (field_1 * field_2).rename(name).assign_attrs(**attrs)


def compute_second_moments(
    ds: xr.Dataset, second_moments: Sequence[Tuple[str, str]],
) -> List[xr.DataArray]:
    """Compute second moments defined using an iterable of tuples.

    Args:
        ds: input Dataset.
        second_moments: iterable of tuples, representing pairs of variable
            names to be combined via products.

    Returns:
        List of DataArrays
    """
    results = []
    for field_1, field_2 in second_moments:
        product = _compute_second_moment(ds[field_1], ds[field_2])
        results.append(product)
    return results


def compute_storage_terms(
    ds: xr.Dataset, storage_terms: Sequence[str], dt: int
) -> List[xr.DataArray]:
    """Compute storage terms from merged dataset.

    Args:
        ds: input Dataset.
        storage_terms: list of variable names
        float: timestep length [seconds].

    Returns:
        List of DataArrays.
    """
    results = []
    for field in storage_terms:
        result = storage(ds[field], dt).rename(f"{field}_storage")
        results.append(result)
    return results


def compute_recoarsened_budget_inputs(
    merged: xr.Dataset,
    dt: int = 15 * 60,
    factor: int = 8,
    first_moments=(
        "T",
        "eddy_flux_vulcan_omega_temp",
        "t_dt_fv_sat_adj_coarse",
        "t_dt_nudge_coarse",
        "t_dt_phys_coarse",
        "sphum",
        "eddy_flux_vulcan_omega_sphum",
        "qv_dt_fv_sat_adj_coarse",
        "qv_dt_phys_coarse",
        "vulcan_omega_coarse",
    ),
    second_moments=(("T", "vulcan_omega_coarse"), ("sphum", "vulcan_omega_coarse")),
    storage_terms=("T", "sphum"),
):
    """Compute the inputs required for the coarse-grained budgets of
    temperature and specific humidity.

    Example output for a single tile::

        <xarray.Dataset>
        Dimensions:                       (grid_xt: 6, grid_yt: 6, pfull: 79)
        Coordinates:
            step                          object ...
            tile                          int64 ...
            time                          object ...
        Dimensions without coordinates: grid_xt, grid_yt, pfull
        Data variables:
            T                             (pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(79, 6, 6), meta=np.ndarray>
            T_storage                     (pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(79, 6, 6), meta=np.ndarray>
            T_vulcan_omega_coarse         (pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(79, 6, 6), meta=np.ndarray>
            delp                          (pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(79, 6, 6), meta=np.ndarray>
            eddy_flux_vulcan_omega_sphum  (pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(79, 6, 6), meta=np.ndarray>
            eddy_flux_vulcan_omega_temp   (pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(79, 6, 6), meta=np.ndarray>
            qv_dt_fv_sat_adj_coarse       (pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(79, 6, 6), meta=np.ndarray>
            qv_dt_phys_coarse             (pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(79, 6, 6), meta=np.ndarray>
            sphum                         (pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(79, 6, 6), meta=np.ndarray>
            sphum_storage                 (pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(79, 6, 6), meta=np.ndarray>
            sphum_vulcan_omega_coarse     (pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(79, 6, 6), meta=np.ndarray>
            t_dt_fv_sat_adj_coarse        (pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(79, 6, 6), meta=np.ndarray>
            t_dt_nudge_coarse             (pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(79, 6, 6), meta=np.ndarray>
            t_dt_phys_coarse              (pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(79, 6, 6), meta=np.ndarray>
            vulcan_omega_coarse           (pfull, grid_yt, grid_xt) float32 dask.array<chunksize=(79, 6, 6), meta=np.ndarray>


    """  # noqa
    logger.info("Re-coarse-graining the fields needed for the fine-resolution budgets")

    middle = merged.sel(step="middle")
    area = middle.area_coarse
    delp_fine = middle.delp
    delp_coarse = GRID.weighted_block_average(delp_fine, area, factor=factor)
    raw_first_moments = [middle[v] for v in first_moments]
    raw_second_moments = compute_second_moments(middle, second_moments)
    raw_storage_terms = compute_storage_terms(merged, storage_terms, dt)
    raw_fields = raw_first_moments + raw_second_moments + raw_storage_terms
    coarsened = coarsen_variables(raw_fields, delp_fine, delp_coarse, area, factor)
    return xr.merge([coarsened, delp_coarse])
