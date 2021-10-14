import fv3fit
import xarray as xr
import numpy as np
import pytest
from fv3fit._shared.models import EnsembleModel


@pytest.mark.parametrize(
    "values, reduction, output",
    [((0.0, 3.0, 5.0), "median", 3.0), ((0.0, 3.0, 5.0), "mean", 8.0 / 3)],
)
def test_ensemble_model_median(values, reduction, output):
    input_variables = ["input"]
    output_variables = ["output"]
    models = tuple(
        fv3fit.testing.ConstantOutputPredictor(input_variables, output_variables)
        for _ in values
    )
    for i, m in enumerate(models):
        m.set_outputs(output=values[i])
    ensemble = EnsembleModel(models, reduction=reduction)
    ds_in = xr.Dataset(
        data_vars={"input": xr.DataArray(np.zeros([3, 3, 5]), dims=["x", "y", "z"],)}
    )
    ds_out = ensemble.predict_columnwise(ds_in, feature_dim="z")
    assert len(ds_out.data_vars) == 1
    assert "output" in ds_out.data_vars
    np.testing.assert_almost_equal(ds_out["output"].values, output)
