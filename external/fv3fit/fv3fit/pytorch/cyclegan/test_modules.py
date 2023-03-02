from fv3fit.pytorch.cyclegan.modules import GeographicFeatures
import torch
import numpy as np
import fv3fit
import pytest


@pytest.mark.parametrize("n_batch", [1, 3], ids=["n_batch=1", "n_batch=3"])
def test_geographic_features_have_correct_range(n_batch: int):
    fv3fit.set_random_seed(0)
    nx = 48
    ny = 48
    n_channel = 2
    module = GeographicFeatures(nx=nx, ny=ny)
    time = torch.rand(n_batch)
    x = torch.rand(n_batch, 6, n_channel, nx, ny)
    y = module((time, x))
    assert y.shape[2] == n_channel + GeographicFeatures.N_FEATURES
    geo_features = y[:, :, n_channel:, :, :]
    # time_x, time_y, and geo x, y, z all have mean 0 and range [-1, 1]
    # as they correspond to evenly spaced grid points on a sphere
    for i in range(GeographicFeatures.N_FEATURES):
        assert np.abs(np.mean(geo_features[:, :, i, :, :].cpu().numpy())) < 1e-3, i
        assert np.abs(geo_features[:, :, i, :, :].cpu().numpy().max() - 1) < 1e-3, i
        assert np.abs(geo_features[:, :, i, :, :].cpu().numpy().min() + 1) < 1e-3, i
