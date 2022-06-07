from toolz import curry
import xarray as xr
import vcm


@curry
def coarsen_cell_centered(
    ds: xr.Dataset, coarsening_factor: int, weights: xr.DataArray
) -> xr.Dataset:
    return vcm.cubedsphere.weighted_block_average(
        ds.drop_vars("area", errors="ignore"),
        weights=weights,
        coarsening_factor=coarsening_factor,
        x_dim="x",
        y_dim="y",
    )


def res_from_string(res_str: str) -> int:
    try:
        return int(res_str.lower().strip("c"))
    except ValueError:
        raise ValueError(
            'res_str must start with "c" or "C" followed by only integers.'
        )
