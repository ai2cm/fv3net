import fv3fit
import xarray as xr
import numpy as np
import pytest
from fv3fit._shared.models import DerivedModel


input_variables = [
    "input",
]
output_variables = [
    "override_for_time_adjusted_total_sky_downward_shortwave_flux_at_surface",
]
base_model = fv3fit.testing.ConstantOutputPredictor(input_variables, output_variables)
base_model.set_outputs(
    override_for_time_adjusted_total_sky_downward_shortwave_flux_at_surface=1.0
)


def test_wrap_another_derived_model():
    base_outputs = [
        "override_for_time_adjusted_total_sky_downward_shortwave_flux_at_surface",
        "dQ2",
    ]
    base_model = fv3fit.testing.ConstantOutputPredictor(["input"], base_outputs)
    base_model.set_outputs(
        override_for_time_adjusted_total_sky_downward_shortwave_flux_at_surface=1.0,
        dQ2=1.0,
    )
    derived_model_0 = DerivedModel(
        base_model, derived_output_variables=["net_shortwave_sfc_flux_derived"],
    )
    derived_model_1 = DerivedModel(derived_model_0, derived_output_variables=["Q2"])
    assert not isinstance(derived_model_1.base_model, DerivedModel)
    assert set(derived_model_1.input_variables) == {
        "input",
        "surface_diffused_shortwave_albedo",
        "pressure_thickness_of_atmospheric_layer",
        "pQ2",
    }
    assert set(derived_model_1.output_variables) == set(base_outputs).union(
        {"Q2", "net_shortwave_sfc_flux_derived"}
    )

    # also need to check that the prediction matches the list of outputs
    arr = xr.DataArray(np.zeros(10), dims=["x"],)
    inputs = xr.Dataset(data_vars={var: arr for var in derived_model_1.input_variables})
    outputs = derived_model_1.predict(inputs)
    assert set(outputs.data_vars) == set(derived_model_1.output_variables)


def test_get_additional_inputs():
    derived_model = DerivedModel(
        base_model, derived_output_variables=["net_shortwave_sfc_flux_derived"],
    )
    # base output for dw shortwave override should not be in the additional inputs
    assert derived_model._additional_input_variables == [
        "surface_diffused_shortwave_albedo"
    ]


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
