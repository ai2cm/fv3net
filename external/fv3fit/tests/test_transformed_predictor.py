import xarray as xr
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
