import xarray as xr
import sys


DIM_RENAME = {
    "x": "grid_xt",
    "y": "grid_yt",
    "xb": "grid_x",
    "yb": "grid_y",
    "x_interface": "grid_x",
    "y_interface": "grid_y",
}

REQUIRED_ATTRS = {
    "grid_xt": {"cartesian_axis": "X"},
    "grid_yt": {"cartesian_axis": "Y"},
    "grid_x": {"cartesian_axis": "X"},
    "grid_y": {"cartesian_axis": "Y"},
}

CUBED_SPHERE_DIMS = {"tile", "grid_xt", "grid_yt"}


if __name__ == "__main__":
    path = sys.argv[1]
    output_prefix = sys.argv[2]

    ds = xr.open_dataset(path)

    existing_dims_to_rename = {k: v for k, v in DIM_RENAME.items() if k in ds.dims}
    ds = ds.rename(existing_dims_to_rename)

    for variable, attrs in REQUIRED_ATTRS.items():
        ds[variable] = ds[variable].assign_attrs(attrs)

    cubed_sphere_variables = [v for v in ds if CUBED_SPHERE_DIMS <= set(ds[v].dims)]
    ds = ds[cubed_sphere_variables]

    for tile in range(6):
        ds.isel(tile=tile).to_netcdf(f"{output_prefix}.tile{tile+1}.nc")
