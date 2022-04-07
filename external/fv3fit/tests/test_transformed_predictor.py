import xarray as xr
import numpy as np
import fv3fit
import vcm

input_variables = [
    "input",
]
output_variables = [
    "Q1",
    "Q2",
]
base_model = fv3fit.testing.ConstantOutputPredictor(input_variables, output_variables)
base_model.set_outputs(Q1=1.0, Q2=2.0)
transforms = [vcm.DataTransform("Qm_from_Q1_Q2")]


def test_transformed_prediction():
    transformed_model = fv3fit.TransformedPredictor(base_model, transforms)
    input_dataset = xr.Dataset({"input": xr.DataArray([0, 1, 2])})
    output = transformed_model.predict(input_dataset)
    assert "Qm" in output


def test_transformed_prediction_additional_required_input():
    transformed_model = fv3fit.TransformedPredictor(
        fv3fit.testing.ConstantOutputPredictor(["input"], ["Q1"]), transforms
    )
    input_dataset = xr.Dataset(
        {"input": xr.DataArray([0, 1, 2]), "Q2": xr.DataArray([0, 1, 2])}
    )
    output = transformed_model.predict(input_dataset)
    assert "Qm" in output
    assert "Q2" not in output


def test_transformed_prediction_when_inputs_contain_an_output():
    # this case can arise in offline diags
    base = fv3fit.testing.ConstantOutputPredictor(["input"], ["Q1"])
    base.set_outputs(Q1=np.array([5, 6, 7]))
    transformed_model = fv3fit.TransformedPredictor(base, transforms)
    input_dataset = xr.Dataset(
        {
            "input": xr.DataArray([0, 1, 2]),
            "Q2": xr.DataArray([0, 1, 2], dims=["z"]),
            "Qm": xr.DataArray([3, 4, 5], dims=["z"]),
        }
    )
    base_output = base.predict(input_dataset)
    merged_output = {"Q1": base_output["Q1"], "Q2": input_dataset["Q2"]}
    expected_Qm = transforms[0].apply(merged_output)["Qm"]
    transformed_model_output = transformed_model.predict(input_dataset)
    assert "Qm" in transformed_model_output
    assert "Q2" not in transformed_model_output
    xr.testing.assert_allclose(expected_Qm, transformed_model_output["Qm"])


def test_transformed_predictor_inputs_outputs():
    transformed_model = fv3fit.TransformedPredictor(base_model, transforms)
    assert transformed_model.input_variables == ["input"]
    assert transformed_model.output_variables == ["Q1", "Q2", "Qm"]


def test_transformed_predictor_inputs_outputs_additional_required_input():
    transformed_model = fv3fit.TransformedPredictor(
        fv3fit.testing.ConstantOutputPredictor(["input"], ["Q1"]), transforms
    )
    assert transformed_model.input_variables == ["Q2", "input"]
    assert transformed_model.output_variables == ["Q1", "Qm"]


def test_dump_and_load(tmpdir):
    transformed_model = fv3fit.TransformedPredictor(base_model, transforms)
    input_dataset = xr.Dataset({"input": xr.DataArray([0, 1, 2])})
    prediction = transformed_model.predict(input_dataset)

    fv3fit.dump(transformed_model, str(tmpdir))
    loaded_model = fv3fit.load(str(tmpdir))

    prediction_after_load = loaded_model.predict(input_dataset)
    assert prediction_after_load.identical(prediction)
