import numpy as np
import xarray as xr

import fv3fit


def _model_dataset() -> xr.Dataset:

    nz = 63
    arr = np.zeros((1, nz))
    arr_1d = np.zeros((1,))
    dims = ["sample", "z"]
    dims_1d = ["sample"]

    data = xr.Dataset(
        {
            "specific_humidity": (dims, arr),
            "air_temperature": (dims, arr),
            "total_sky_downward_shortwave_flux_at_surface": (dims_1d, arr_1d),
            "total_sky_downward_longwave_flux_at_surface": (dims_1d, arr_1d),
            "dQ1": (dims, arr),
            "dQ2": (dims, arr),
            "dQu": (dims, arr),
            "dQv": (dims, arr),
        }
    )

    return data


def get_mock_predictor(
    model_predictands: str = "tendencies", dQ1_tendency=0.0
) -> fv3fit.Predictor:

    data = _model_dataset()
    nz = data.sizes["z"]
    if model_predictands == "tendencies":
        output_variables = ["dQ1", "dQ2", "dQu", "dQv"]
    elif model_predictands == "rad_fluxes":
        output_variables = [
            "total_sky_downward_shortwave_flux_at_surface",
            "total_sky_downward_longwave_flux_at_surface",
        ]
    elif model_predictands == "state_and_tendency":
        output_variables = ["dQ1", "total_sky_downward_shortwave_flux_at_surface"]
    elif model_predictands == "state_tendency_and_intermediate":
        output_variables = [
            "dQ1",
            "total_sky_downward_shortwave_flux_at_surface",
            "shortwave_transmissivity_of_atmospheric_column",
        ]
    outputs = {
        "dQ1": np.full(nz, dQ1_tendency),
        # include nonzero moistening to test for mass conservation
        "dQ2": np.full(nz, -1e-4 / 86400),
        "dQu": np.full(nz, 1 / 86400),
        "dQv": np.full(nz, 1 / 86400),
        "total_sky_downward_shortwave_flux_at_surface": 300.0,
        "total_sky_downward_longwave_flux_at_surface": 400.0,
        "shortwave_transmissivity_of_atmospheric_column": 0.5,
    }
    predictor = fv3fit.testing.ConstantOutputPredictor(
        input_variables=["air_temperature", "specific_humidity"],
        output_variables=output_variables,
    )
    predictor.set_outputs(**outputs)

    return predictor
