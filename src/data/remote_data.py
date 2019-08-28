import xarray as xr


def merge_intake_xarray_sets(catalog, entries):
    datasets = [catalog[entry].to_dask() for entry in entries]
    return xr.merge(datasets)


def open_gfdl_data(catalog):
    entries_to_merge = [
        '2019-08-12-james-huff-additional-gfdl-fields',
        '2019-07-17-GFDL_FV3_DYAMOND_0.25deg_3hr_3d',
    ]
    return merge_intake_xarray_sets(catalog, entries_to_merge).pipe(remove_pressure_level_variables)


def remove_pressure_level_variables(ds):
    variables = [field for field in ds.data_vars if not field.endswith('plev')]
    return ds[variables]