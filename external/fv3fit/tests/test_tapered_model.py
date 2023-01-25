import numpy as np
import os
import xarray as xr
import yaml

from fv3fit._shared.config import TaperConfig
from fv3fit._shared.models import TaperedModel

from fv3fit.testing import ConstantOutputPredictor


def test_TaperedModel():
    model = ConstantOutputPredictor(
        input_variables=["in0", "in1"], output_variables=["out0", "out1"]
    )
    model.set_outputs(out1=np.ones(10), out0=np.ones(10))
    taper_config0 = TaperConfig(cutoff=3, rate=5.0, taper_dim="z")
    taper_config1 = TaperConfig(cutoff=6, rate=3.0, taper_dim="z")
    tapered_model = TaperedModel(model, {"out0": taper_config0, "out1": taper_config1})
    da = xr.DataArray(data=np.ones((5, 10)), dims=["x", "z"])
    X = xr.Dataset({"in0": da, "in1": da})

    tapered_prediction = tapered_model.predict(X)
    np.testing.assert_array_equal(
        tapered_prediction["out0"].values, taper_config0.apply(model.predict(X)["out0"])
    )
    np.testing.assert_array_equal(
        tapered_prediction["out1"].values, taper_config1.apply(model.predict(X)["out1"])
    )


def test_TaperedModel_load(tmpdir):

    model = ConstantOutputPredictor(
        input_variables=["in0", "in1"], output_variables=["out0", "out1"]
    )
    model.set_outputs(out1=np.ones(10), out0=np.ones(10))

    base_model_output_path = f"{str(tmpdir)}/predictor"
    os.mkdir(base_model_output_path)
    model.dump(base_model_output_path)
    config = {
        "tapering": {
            "out0": {"cutoff": 3, "rate": 5},
            "out1": {"cutoff": 2, "rate": 6},
        },
        "model": base_model_output_path,
    }
    output_path = f"{str(tmpdir)}/tapered_model"
    os.mkdir(output_path)
    with open(f"{output_path}/tapered_model.yaml", "w") as f:
        yaml.dump(config, f)
    with open(f"{output_path}/name", "w") as f:
        print("tapered_model", file=f)

    tapered_model = TaperedModel.load(output_path)
    da = xr.DataArray(data=np.ones((5, 10)), dims=["x", "z"])
    X = xr.Dataset({"in0": da, "in1": da})
    tapered_prediction = tapered_model.predict(X)
    assert np.mean(tapered_prediction["out0"]) < 1.0
