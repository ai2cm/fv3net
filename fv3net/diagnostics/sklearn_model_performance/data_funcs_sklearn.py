import fsspec
from scipy.interpolate import UnivariateSpline
import xarray as xr

from vcm.calc import mass_integrate, thermo
from vcm.convenience import round_time
from vcm.cubedsphere.constants import (
    INIT_TIME_DIM,
    COORD_X_CENTER,
    COORD_Y_CENTER,
    TILE_COORDS,
)
from vcm.regrid import regrid_to_shared_coords


kg_m2s_to_mm_day = (1e3 * 86400) / 997.0
kg_m2_to_mm = 1000.0 / 997

p0 = 100000  # reference pressure for potential temp [Pa]
SPECIFIC_HEAT_CONST_PRESSURE = 1004  # [J/kg K]
GRAVITY = 9.81  # [m/s2]
POISSON_CONST = 0.2854
SEC_PER_DAY = 86400

SAMPLE_DIM = "sample"
STACK_DIMS = ["tile", INIT_TIME_DIM, COORD_X_CENTER, COORD_Y_CENTER]


def predict_on_test_data(test_data_path, model_path, num_test_zarrs, model_type="rf"):
    if model_type == "rf":
        from fv3net.regression.sklearn.test import (
            load_test_dataset,
            load_model,
            predict_dataset,
        )

        ds_test = load_test_dataset(test_data_path, num_test_zarrs)
        sk_wrapped_model = load_model(model_path)
        ds_pred = predict_dataset(sk_wrapped_model, ds_test)
        return ds_test.unstack(), ds_pred
    else:
        raise ValueError(
            "Cannot predict using model type {model_type},"
            "only 'rf' is currently implemented."
        )


def load_high_res_diag_dataset(coarsened_hires_diags_path, init_times):
    fs = fsspec.filesystem("gs")
    ds_hires = xr.open_zarr(
        fs.get_mapper(coarsened_hires_diags_path), consolidated=True
    ).rename({"time": INIT_TIME_DIM})
    ds_hires = ds_hires.assign_coords(
        {
            INIT_TIME_DIM: [round_time(t) for t in ds_hires[INIT_TIME_DIM].values],
            "tile": TILE_COORDS,
        }
    )
    ds_hires = ds_hires.sel({INIT_TIME_DIM: list(set(init_times))})
    if set(ds_hires[INIT_TIME_DIM].values) != set(init_times):
        raise ValueError(
            f"Timesteps {set(init_times)-set(ds_hires[INIT_TIME_DIM].values)}"
            f"are not matched in high res dataset."
        )

    evaporation = thermo.latent_heat_flux_to_evaporation(ds_hires["LHTFLsfc_coarse"])
    ds_hires["P-E"] = SEC_PER_DAY * (ds_hires["PRATEsfc_coarse"] - evaporation)
    ds_hires["heating"] = thermo.net_heating_from_dataset(ds_hires)
    return ds_hires


def add_integrated_Q_vars(ds):
    """ Integrates column Q1, Q2 to add variables heating and P-E. Modifies in place.
    
    Args:
        ds (xarray dataset): dataset that has Q1, Q2, delp data variables
    """
    ds["P-E"] = mass_integrate(-ds["Q2"], ds.delp) * kg_m2s_to_mm_day
    ds["heating"] = SPECIFIC_HEAT_CONST_PRESSURE * mass_integrate(ds["Q1"], ds.delp)


def integrate_for_Q(P, sphum, lower_bound=55000, upper_bound=85000):
    spline = UnivariateSpline(P, sphum)
    return (spline.integral(lower_bound, upper_bound) / GRAVITY) * kg_m2_to_mm


def lower_tropospheric_stability(ds):
    pressure = thermo.pressure_at_midpoint_log(ds.delp)
    T_at_700mb = (
        regrid_to_shared_coords(
            ds["T"],
            [70000],
            pressure,
            regrid_dim_name="p700mb",
            replace_dim_name="pfull",
        )
        .squeeze()
        .drop("p700mb")
    )
    theta_700mb = thermo.potential_temperature(70000, T_at_700mb)
    return theta_700mb - ds["tsea"]
