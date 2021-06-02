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
            "downward_shortwave": (dims_1d, arr_1d),
            "net_shortwave": (dims_1d, arr_1d),
            "downward_longwave": (dims_1d, arr_1d),
            "dQ1": (dims, arr),
            "dQ2": (dims, arr),
            "dQu": (dims, arr),
            "dQv": (dims, arr),
        }
    )

    return data


def get_mock_predictor(model_predictands: str = "tendencies") -> fv3fit.Predictor:

    data = _model_dataset()
    nz = data.sizes["z"]
    if model_predictands == "tendencies":
        output_variables = ["dQ1", "dQ2", "dQu", "dQv"]
    elif model_predictands == "rad_fluxes":
        output_variables = ["downward_shortwave", "net_shortwave", "downward_longwave"]
    outputs = {
        "dQ1": np.zeros(nz),
        "dQ2": -np.full(nz, 1e-4 / 86400),
        "dQu": np.full(nz, 1 / 86400),
        "dQv": np.full(nz, 1 / 86400),
        "downward_shortwave": 300.0,
        "net_shortwave": 250.0,
        "downward_longwave": 400.0,
    }
    predictor = fv3fit.testing.ConstantOutputPredictor(
        sample_dim_name="sample",
        input_variables=["air_temperature", "specific_humidity"],
        output_variables=output_variables,
    )
    predictor.set_outputs(**outputs)

    return predictor
