from typing import Iterable, Hashable
import fv3fit
import xarray as xr
import numpy as np
import pytest
from fv3fit._shared.models import EnsembleModel


class ConstantModel(fv3fit.Predictor):
    """Model with constant outputs."""

    def __init__(
        self,
        sample_dim_name: str,
        input_variables: Iterable[Hashable],
        output_variables: Iterable[Hashable],
        output_value: float = 0.0,
    ):
        """Initialize the predictor
        
        Args:
            sample_dim_name: name of sample dimension
            input_variables: names of input variables
            output_variables: names of output variables
        
        """
        self.output_value = output_value
        super().__init__(sample_dim_name, input_variables, output_variables)

    def load(cls, path):
        raise NotImplementedError()

    def predict(self, X: xr.Dataset):
        input_da = X[self.input_variables[0]]
        data_vars = {}
        for name in self.output_variables:
            da = input_da.copy()
            da[:] = self.output_value
            data_vars[name] = da
        return xr.Dataset(data_vars)


@pytest.mark.parametrize(
    "values, reduction, output",
    [((0.0, 3.0, 5.0), "median", 3.0), ((0.0, 3.0, 5.0), "mean", 8.0 / 3)],
)
def test_ensemble_model_median(values, reduction, output):
    sample_dim_name = "sample"
    input_variables = ["input"]
    output_variables = ["output"]
    models = tuple(
        ConstantModel(sample_dim_name, input_variables, output_variables, value)
        for value in values
    )
    ensemble = EnsembleModel(models, reduction=reduction)
    ds_in = xr.Dataset(
        data_vars={"input": xr.DataArray(np.zeros([3, 3, 5]), dims=["x", "y", "z"],)}
    )
    ds_out = ensemble.predict(ds_in)
    assert len(ds_out.data_vars) == 1
    assert "output" in ds_out.data_vars
    np.testing.assert_almost_equal(ds_out["output"].values, output)
