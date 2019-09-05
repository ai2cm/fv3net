import xarray as xr
import intake


def merge_intake_xarray_sets(catalog, entries):
    datasets = [catalog[entry].to_dask() for entry in entries]
    return xr.merge(datasets)


def open_gfdl_data(catalog):
    entries_to_merge = [
        "2019-08-12-james-huff-additional-gfdl-fields",
        "2019-07-17-GFDL_FV3_DYAMOND_0.25deg_3hr_3d",
    ]
    return merge_intake_xarray_sets(catalog, entries_to_merge).pipe(
        remove_pressure_level_variables
    )


def open_gfdl_data_with_2d(catalog: intake.Catalog) -> xr.Dataset:
    """Open the initial DYAMOND prototype data merging the 2D and 3D data

    The snapshot of the 2D data closest to the 3D time output time is included.
    The 2D data are not averaged in time.
    """
    key_2d = "2019-07-17-GFDL_FV3_DYAMOND_0.25deg_15minute_2d"
    data_2d = catalog[key_2d].to_dask()
    data_3d = open_gfdl_data(catalog)
    return xr.merge([data_3d, data_2d], join="left")


def remove_pressure_level_variables(ds):
    variables = [field for field in ds.data_vars if not field.endswith("plev")]
    return ds[variables]
