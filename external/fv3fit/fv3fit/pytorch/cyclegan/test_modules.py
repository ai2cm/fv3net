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
    SECONDS_PER_DAY = 24 * 60 * 60
    time = torch.rand(n_batch) * SECONDS_PER_DAY
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


#    # The following code can be uncommented to visualize the geographic features
#    # as cross plots. Useful during development.

#     plot_cross(
#         [
#             geo_features[0, :, i, :, :].cpu().numpy()
#             for i in range(geo_features.shape[2])
#         ],
#         "0.0",
#     )
#     SECONDS_PER_DAY = 24 * 60 * 60
#     time[:] = 0.25 * SECONDS_PER_DAY
#     x = torch.rand(n_batch, 6, n_channel, nx, ny)
#     y = module((time, x))
#     geo_features = y[:, :, n_channel:, :, :]
#     plot_cross(
#         [
#             geo_features[0, :, i, :, :].cpu().numpy()
#             for i in range(geo_features.shape[2])
#         ],
#         "0.5",
#     )


# def plot_cross(
#     data_list, label: str,
# ):
#     import xarray as xr
#     import matplotlib.pyplot as plt
#     from vcm.cubedsphere import to_cross

#     """
#     Plot global states as cross-plots.

#     Args:
#         real_a: Real state from domain A, shape [tile, x, y]
#         real_b: Real state from domain B, shape [tile, x, y]
#         fake_a: Fake state from domain A, shape [tile, x, y]
#         fake_b: Fake state from domain B, shape [tile, x, y]

#     Returns:
#         io.BytesIO: BytesIO object containing the plot
#     """

#     fig, ax = plt.subplots(len(data_list), 1, figsize=(7, 3.5 * len(data_list)))
#     for i, data in enumerate(data_list):
#         to_cross(
#             xr.DataArray(data, dims=["tile", "grid_xt", "grid_yt"])
#         ).plot(ax=ax[i])
#     plt.tight_layout()
#     plt.savefig(f"test_geographic_features_have_correct_range_{label}.png")
