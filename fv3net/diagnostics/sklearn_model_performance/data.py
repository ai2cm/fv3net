import numpy as np
import os
from scipy.interpolate import UnivariateSpline
import xarray as xr

import fv3net
from ..data import net_heating_from_dataset

import vcm
from vcm.cloud.fsspec import get_fs
from vcm.convenience import round_time
from vcm.regrid import regrid_to_shared_coords
from vcm.constants import (
    kg_m2s_to_mm_day,
    kg_m2_to_mm,
    SPECIFIC_HEAT_CONST_PRESSURE,
    GRAVITY,
)
import logging

logger = logging.getLogger(__file__)

SAMPLE_DIM = "sample"

THERMO_DATA_VAR_ATTRS = {
    "net_precipitation": {"long_name": "net column precipitation", "units": "mm/day"},
    "net_heating": {"long_name": "net column heating", "units": "W/m^2"},
    "net_precipitation_ml": {
        "long_name": "residual P-E predicted by ML model",
        "units": "mm/day",
    },
    "net_heating_ml": {
        "long_name": "residual heating predicted by ML model",
        "units": "W/m^2",
    },
}


def predict_on_test_data(
    test_data_path,
    model_path,
    pred_vars_to_keep,
    init_time_dim="initial_time",
    coord_z_center="z",
    model_type="rf",
):

    if model_type == "rf":
        from fv3net.regression.sklearn.test import (
            load_test_dataset,
            load_model,
            predict_dataset,
        )

        ds_test = load_test_dataset(test_data_path, init_time_dim, coord_z_center)
        sk_wrapped_model = load_model(model_path)
        logger.info("Making prediction with sklearn model")
        ds_pred = predict_dataset(sk_wrapped_model, ds_test, pred_vars_to_keep)
        return ds_test.unstack(), ds_pred
    else:
        raise ValueError(
            "Cannot predict using model type {model_type},"
            "only 'rf' is currently implemented."
        )


def load_high_res_diag_dataset(
    coarsened_hires_diags_path, init_times, init_time_dim, renamed_hires_grid_vars
):
    logger.info("load_high_res_diag_dataset")
    fs = get_fs(coarsened_hires_diags_path)
    ds_hires = xr.open_zarr(
        # fs.get_mapper functions like a zarr store
        fs.get_mapper(
            os.path.join(coarsened_hires_diags_path, fv3net.COARSENED_DIAGS_ZARR_NAME)
        ),
        consolidated=True,
    ).rename({"time": init_time_dim, **renamed_hires_grid_vars})
    ds_hires = ds_hires.assign_coords(
        {
            init_time_dim: [round_time(t) for t in ds_hires[init_time_dim].values],
            "tile": range(6),
        }
    )
    ds_hires = ds_hires.sel({init_time_dim: list(set(init_times))})
    if set(ds_hires[init_time_dim].values) != set(init_times):
        raise ValueError(
            f"Timesteps {set(init_times)-set(ds_hires[init_time_dim].values)}"
            f"are not matched in high res dataset."
        )

    ds_hires["net_precipitation"] = vcm.net_precipitation(
        ds_hires[f"LHTFLsfc_coarse"], ds_hires[f"PRATEsfc_coarse"]
    )
    ds_hires["net_heating"] = net_heating_from_dataset(ds_hires, suffix="coarse")

    return ds_hires
    

def total_Q(ds, vars=["Q1", "Q2"]):
    for var in vars:
        ds[var] = ds[f"d{var}"] + ds[f"p{var}"]
    return ds
    

def add_column_heating_moistening(
    ds,
    suffix_coarse_train_diag,
    var_pressure_thickness,
    var_q_moistening_ml,
    var_q_heating_ml,
    coord_z_center,
):
    """ Integrates column dQ1, dQ2 and sum with model's heating/moistening to calculate
    heating and P-E. Modifies in place.
    
    Args:
        ds (xarray dataset): train/test or prediction dataset
            that has dQ1, dQ2, delp, precip and LHF data variables
    """
    logger.info("add_column_heating_moistening")

    ds["net_precipitation_ml"] = (
        vcm.mass_integrate(
            -ds[var_q_moistening_ml], ds[var_pressure_thickness], dim=coord_z_center
        )
        * kg_m2s_to_mm_day
    )
    if (
        f"LHTFLsfc_{suffix_coarse_train_diag}" in ds.data_vars
        and f"PRATEsfc_{suffix_coarse_train_diag}" in ds.data_vars
    ):
        ds["net_precipitation_physics"] = vcm.net_precipitation(
            ds[f"LHTFLsfc_{suffix_coarse_train_diag}"],
            ds[f"PRATEsfc_{suffix_coarse_train_diag}"],
        )
        ds["net_precipitation"] = (
            ds["net_precipitation_ml"] + ds["net_precipitation_physics"]
        )
    else:
        # fill in zeros for physics values if all physics off configured data
        ds = _fill_zero_da_from_template(
            ds, "net_precipitation_physics", ds["net_precipitation_ml"]
        )
        ds["net_precipitation"] = ds["net_precipitation_ml"]

    ds["net_heating_ml"] = SPECIFIC_HEAT_CONST_PRESSURE * vcm.mass_integrate(
        ds[var_q_heating_ml], ds[var_pressure_thickness], dim=coord_z_center
    )
    if f"SHTFLsfc_{suffix_coarse_train_diag}" in ds.data_vars:
        ds["net_heating_physics"] = net_heating_from_dataset(
            ds, suffix=suffix_coarse_train_diag
        )
        ds["net_heating"] = ds["net_heating_ml"] + ds["net_heating_physics"]
    else:
        # fill in zeros for physics values if all physics off configured data
        ds = _fill_zero_da_from_template(
            ds, "net_heating_physics", ds["net_heating_ml"]
        )
        ds["net_heating"] = ds["net_heating_ml"]

    for data_var, data_attrs in THERMO_DATA_VAR_ATTRS.items():
        ds[data_var].attrs = data_attrs

    return ds


def _fill_zero_da_from_template(ds, zero_da_name, template_dataarray):
    da_fill = np.empty(template_dataarray.shape)
    da_fill[:] = 0.0
    return ds.assign({zero_da_name: (template_dataarray.dims, da_fill)})


def integrate_for_Q(P, sphum, lower_bound=55000, upper_bound=85000):
    spline = UnivariateSpline(P, sphum)
    return (spline.integral(lower_bound, upper_bound) / GRAVITY) * kg_m2_to_mm


def lower_tropospheric_stability(da_T, da_delp, da_Tsfc, coord_z_center="z"):
    pressure = vcm.pressure_at_midpoint_log(da_delp, dim=coord_z_center)
    T_at_700mb = (
        regrid_to_shared_coords(
            da_T,
            [70000],
            pressure,
            regrid_dim_name="p700mb",
            replace_dim_name=coord_z_center,
        )
        .squeeze()
        .drop("p700mb")
    )
    theta_700mb = vcm.potential_temperature(70000, T_at_700mb)
    return theta_700mb - da_Tsfc
