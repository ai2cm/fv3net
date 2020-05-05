import .calc


"""
Create *dataset functions organize the diagnostics into a single dataset to be returned and saved
Only calculations that go on in here are aggregations/means, all other calculations to derive
quantities shown in diagnostics should already be done before these get called.
"""


def create_diagnostics_dataset(data_arrays, config) -> xr.Dataset:
    """
    below are the variable type: dims that are saved, 
    in addition there is also a source dataset coord (target, hires, pred)

    time avg'd mapped variables: grid dims
    snapshotted mapped variables: grid dims + time
    vertical profiled variables: pressure dim 

    diurnal cycle variables: binned local time dim
    """
    # example- makes time avg maps of variables listed under that 
    # diagnostic in the config, and adds those to the diagnostics dataset
    ds = ds.merge(_dataset_from_config_entry(_map_time_avg, data_arrays, config["time_avg_maps"]))
    
    pass


def _dataset_from_config_entry(diag_func, data_arrays, config_entry):
    """
    config entry is a single diagnostic type (e.g. time avg map) in the config yaml, 
    which contains list of variables to calculate diagnostic for , ex.
    - time_avg_maps:
        net_precipitation_total:
            sources:
            - target
            - pred
            - hires
        net_heating_total:
            sources:
            - target
            - pred
            - hires
    diag_func gets called on the data arrays to create each diagnostic
    """
    ds = xr.Dataset()
    for var in config_entry:
        diag_data_arrays = [data_arrays[f"{var}_{source}"] for source in config_entry["sources"]]
        diag = [diag_func(diag_data_arrays, config_entry["sources"])]
        # e.g. diags = [_map_time_avg(diag_data_arrays, config_entry["sources"])]
        ds.merge(diag)
    return ds


def create_lower_trop_stability_dataset(data_arrays):
    """    
    lower tropospheric stability: plot dims are Q and net prceip
    this one doesn't fit in with the rest of the diagnostics' dimensions
    so might be easier to save as its own small datasets
    """
    pass


def _map_time_avg(data_arrays: List[xr.DataArray], source_coords: List[str]) -> xr.Dataset:
    """
    for all the variables in the configuration
    time averages data arrays and returns them merged into a dataset with the source coord
    """
    pass

