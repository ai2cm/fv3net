from fv3fit.keras._models.shared.utils import count_features
import xarray as xr
import numpy as np
import pytest


@pytest.mark.parametrize("n_dims", [1, 3])
def test_count_features_no_feature_dim(n_dims: int):
    dims = [f"dim_{i}" for i in range(n_dims)]
    ds = xr.Dataset(
        data_vars={
            "var": xr.DataArray(np.zeros(tuple(4 for _ in range(n_dims))), dims=dims,)
        }
    )
    features = count_features(["var"], batch=ds, sample_dims=dims)
    assert features["var"] == 1


@pytest.mark.parametrize("n_features", [1, 10])
@pytest.mark.parametrize("n_sample_dims", [1, 3])
def test_count_features_one_feature_dim(n_sample_dims: int, n_features: int):
    sample_dims = [f"dim_{i}" for i in range(n_sample_dims)]
    dims = sample_dims + ["feature"]
    shape = list(4 for _ in range(n_sample_dims)) + [n_features]
    ds = xr.Dataset(data_vars={"var": xr.DataArray(np.zeros(shape), dims=dims,)})
    features = count_features(["var"], batch=ds, sample_dims=sample_dims)
    assert features["var"] == n_features
