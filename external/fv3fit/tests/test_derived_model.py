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


def test_get_additional_inputs():
    derived_model = DerivedModel(
        base_model, derived_output_variables=["net_shortwave_sfc_flux_derived"],
    )
    assert (
        "surface_diffused_shortwave_albedo" in derived_model._additional_input_variables
    )


def test_derived_prediction():
    derived_model = DerivedModel(
        base_model, derived_output_variables=["net_shortwave_sfc_flux_derived"],
    )
    ds_in = xr.Dataset(
        data_vars={
            "input": xr.DataArray(np.zeros([3, 3, 5]), dims=["x", "y", "z"],),
            "surface_diffused_shortwave_albedo": xr.DataArray(
                np.zeros([3, 3]), dims=["x", "y"],
            ),
        }
    )
    prediction = derived_model.predict(ds_in)
    assert "net_shortwave_sfc_flux_derived" in prediction


def test_derived_alert_to_missing_additional_input():
    derived_model = DerivedModel(
        base_model, derived_output_variables=["net_shortwave_sfc_flux_derived"],
    )
    ds_in = xr.Dataset(
        data_vars={"input": xr.DataArray(np.zeros([3, 3, 5]), dims=["x", "y", "z"])}
    )
    with pytest.raises(KeyError):
        derived_model.predict(ds_in.stack(sample=["x", "y"]))


def test_invalid_derived_output_variables():
    with pytest.raises(ValueError):
        DerivedModel(
            base_model, derived_output_variables=["variable_not_in_DerivedMapping"],
        )


def test_dump_and_load(tmpdir):
    derived_model = DerivedModel(
        base_model, derived_output_variables=["net_shortwave_sfc_flux_derived"],
    )
    ds_in = xr.Dataset(
        data_vars={
            "input": xr.DataArray(np.zeros([3, 3, 5]), dims=["x", "y", "z"],),
            "surface_diffused_shortwave_albedo": xr.DataArray(
                np.zeros([3, 3]), dims=["x", "y"],
            ),
        }
    )
    prediction = derived_model.predict(ds_in)

    fv3fit.dump(derived_model, str(tmpdir))
    loaded_model = fv3fit.load(str(tmpdir))

    prediction_after_load = loaded_model.predict(ds_in)
    assert prediction_after_load.identical(prediction)
