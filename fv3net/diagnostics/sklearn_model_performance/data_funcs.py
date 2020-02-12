import fsspec
import xarray as xr

from vcm.convenience import round_time
from vcm.cubedsphere.constants import (
    INIT_TIME_DIM,
    COORD_X_CENTER,
    COORD_Y_CENTER,
    TILE_COORDS,
)
from vcm.calc.thermo import LATENT_HEAT_VAPORIZATION
from ..data_funcs import energy_convergence

kg_m2s_to_mm_day = (1e3 * 86400) / 997.0

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


def load_high_res_diag_dataset(coarsened_hires_diags_path, init_times=None):
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
    if init_times:
        ds_hires = ds_hires.sel({INIT_TIME_DIM: list(set(init_times))})
        if set(ds_hires[INIT_TIME_DIM].values) != set(init_times):
            raise ValueError(
                f"Timesteps {set(init_times)-set(ds_hires[INIT_TIME_DIM].values)}"
                f"are not matched in high res dataset."
            )
    ds_hires["P-E"] = SEC_PER_DAY * (
        ds_hires["PRATEsfc_coarse"]
        - ds_hires["LHTFLsfc_coarse"] / LATENT_HEAT_VAPORIZATION
    )
    ds_hires["heating"] = energy_convergence(ds_hires)
    return ds_hires
