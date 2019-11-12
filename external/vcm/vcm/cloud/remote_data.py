import xarray as xr
import intake
import gcsfs


def merge_intake_xarray_sets(catalog, entries):
    datasets = [catalog[entry].to_dask() for entry in entries]
    return xr.merge(datasets)


def open_gfdl_data(catalog):
    """Test"""
    entries_to_merge = [
        "2019-08-12-james-huff-additional-gfdl-fields",
        "2019-07-17-GFDL_FV3_DYAMOND_0.25deg_3hr_3d",
    ]
    return merge_intake_xarray_sets(catalog, entries_to_merge).pipe(
        _remove_pressure_level_variables
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


def open_gfdl_15_minute_SHiELD(catalog: intake.Catalog, dataset_name: str) -> xr.Dataset:
    """Open the initial SHiELD prototype data."""

    # TODO: combine with function above and generalize
    dset = catalog[dataset_name].to_dask()

    # Change names in SHiELD to names used from DYAMOND
    rename_list = {
        'ucomp': 'u',
        'vcomp': 'v',
        'sphum': 'qv',
        'HGTsfc': 'zs',
        'delz': 'dz',
        'delp': 'dp'
    }

    for old_varname, new_varname in rename_list.items():
        if old_varname in dset:
            # TODO: debug statment about variable renaming taking place
            dset = dset.rename({old_varname: new_varname})
    return dset


def _remove_pressure_level_variables(ds):
    variables = [field for field in ds.data_vars if not field.endswith("plev")]
    return ds[variables]


def write_cloud_zarr(ds, gcs_path):
    fs = gcsfs.GCSFileSystem(project='vcm-ml')
    mapping = fs.get_mapper(gcs_path)
    ds.to_zarr(store=mapping, mode='w')
    return ds
