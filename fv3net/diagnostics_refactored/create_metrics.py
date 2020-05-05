def create_metrics_dataset(data_arrays) -> xr.Dataset:
    ... 
    delp = data_arrays["pressure_thickness_of_atmospheric_layers"] # or whatever the name was
    # example of data_vars dict entries as args in an individual metric function
    ds_metrics = ds_metrics.assign({
        "rmse_dQ1_pressure_level", 
        _calc_rmse_pressure_levels(
            data_arrays["dQ1_target"], data_arrays["dQ1_pred"], delp)
    })
    pass


def _calc_rmse_pressure_levels(
    da_test: xr.DataArray, da_pred: xr.DataArray, delp: xr.DataArray) -> xr.DataArray:
    """
    example of a calc function that takes data arrays as input
    """

    pass