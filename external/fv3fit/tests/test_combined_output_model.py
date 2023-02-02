import numpy as np
import os
import pytest
import xarray as xr
import yaml


from fv3fit._shared.models import CombinedOutputModel
from fv3fit.testing import ConstantOutputPredictor


def test_CombinedOutputModel():
    model0 = ConstantOutputPredictor(
        input_variables=["in0", "in1"], output_variables=["out0a", "out0b"]
    )
    model1 = ConstantOutputPredictor(
        input_variables=["in1", "in2"], output_variables=["out1a", "out1b"]
    )

    model0.set_outputs(out0a=np.ones(10), out0b=np.ones(10))
    model1.set_outputs(out1a=np.ones(10) * 2, out1b=np.ones(10) * 2)

    da = xr.DataArray(data=np.ones((5, 10)), dims=["x", "z"])
    X = xr.Dataset({"in0": da, "in1": da, "in2": da})

    combined_model = CombinedOutputModel([model0, model1])
    assert set(combined_model.input_variables) == {"in0", "in1", "in2"}
    assert set(combined_model.output_variables) == {"out0a", "out0b", "out1a", "out1b"}

    combined_prediction = combined_model.predict(X)

    np.testing.assert_array_equal(
        combined_prediction["out0a"], model0.predict(X)["out0a"]
    )
    np.testing.assert_array_equal(
        combined_prediction["out1a"], model1.predict(X)["out1a"]
    )


def test_CombinedOutputModel_load(tmpdir):
    model0 = ConstantOutputPredictor(
        input_variables=["in0", "in1"], output_variables=["out0a", "out0b"]
    )
    model1 = ConstantOutputPredictor(
        input_variables=["in1", "in2"], output_variables=["out1a", "out1b"]
    )

    model0.set_outputs(out0a=np.ones(10), out0b=np.ones(10))
    model1.set_outputs(out1a=np.ones(10) * 2, out1b=np.ones(10) * 2)

    config = {"models": []}

    for i, model in enumerate([model0, model1]):
        base_model_output_path = f"{str(tmpdir)}/predictor{i}"
        os.mkdir(base_model_output_path)
        model.dump(base_model_output_path)
        config["models"].append(base_model_output_path)

    output_path = f"{str(tmpdir)}/combined_output_model"
    os.mkdir(output_path)
    with open(f"{output_path}/combined_output_model.yaml", "w") as f:
        yaml.dump(config, f)
    with open(f"{output_path}/name", "w") as f:
        print("combined_output_model", file=f)

    da = xr.DataArray(data=np.ones((5, 10)), dims=["x", "z"])
    X = xr.Dataset({"in0": da, "in1": da, "in2": da})

    combined_model = CombinedOutputModel.load(output_path)
    combined_prediction = combined_model.predict(X)
    assert {"out0a", "out0b", "out1a", "out1b"} == set(combined_prediction.data_vars)


def test_error_on_duplicate_outputs():
    model0 = ConstantOutputPredictor(
        input_variables=["in0", "in1"], output_variables=["out0a", "out0b"]
    )
    model1 = ConstantOutputPredictor(
        input_variables=["in1", "in2"], output_variables=["out0a"]
    )
    with pytest.raises(ValueError):
        CombinedOutputModel([model0, model1])
