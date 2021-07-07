import fv3fit
import xarray as xr
import numpy as np
import pytest
from fv3fit._shared.models import DerivedModel


sample_dim_name = "sample"
input_variables = [
    "input",
]
output_variables = [
    "override_for_time_adjusted_total_sky_downward_shortwave_flux_at_surface",
]
base_model = fv3fit.testing.ConstantOutputPredictor(
    sample_dim_name, input_variables, output_variables
)
base_model.set_outputs(
    override_for_time_adjusted_total_sky_downward_shortwave_flux_at_surface=1.0
)


def test_derived_prediction():
    derived_model = DerivedModel(
        base_model,
        additional_input_variables=["surface_diffused_shortwave_albedo"],
        derived_output_variables=["net_shortwave_sfc_flux_derived"],
    )
    ds_in = xr.Dataset(
        data_vars={
            "input": xr.DataArray(np.zeros([3, 3, 5]), dims=["x", "y", "z"],),
            "surface_diffused_shortwave_albedo": xr.DataArray(
                np.zeros([3, 3]), dims=["x", "y"],
            ),
        }
    )
    prediction = derived_model.predict(ds_in.stack(sample=["x", "y"]))
    assert "net_shortwave_sfc_flux_derived" in prediction


def test_derived_alert_to_missing_additional_input():
    derived_model = DerivedModel(
        base_model,
        additional_input_variables=["surface_diffused_shortwave_albedo"],
        derived_output_variables=["net_shortwave_sfc_flux_derived"],
    )
    ds_in = xr.Dataset(
        data_vars={"input": xr.DataArray(np.zeros([3, 3, 5]), dims=["x", "y", "z"])}
    )
    with pytest.raises(KeyError):
        derived_model.predict(ds_in.stack(sample=["x", "y"]))
