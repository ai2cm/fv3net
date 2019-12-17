from fv3net.pipelines.create_training_data.dataset_creator import _create_train_cols

import pytest
import xarray as xr

@pytest.fixture
def test_training_raw_ds():
    centered_coords = {
        "grid_yt": [1], "grid_xt": [1],
        "initialization_time": [np.datetime64('2020-01-01T00:00').astype('M8[ns]'),
                                np.datetime64('2020-01-01T00:15').astype('M8[ns]')],
        "forecast_time": [np.datetime64('2020-01-01T00:00').astype('M8[ns]'),
                          np.datetime64('2020-01-01T00:15').astype('M8[ns]')]}
    x_component_edge_coords = {
        "grid_y": [0, 2], "grid_xt": [1],
        "initialization_time": [np.datetime64('2020-01-01T00:00').astype('M8[ns]'),
                                np.datetime64('2020-01-01T00:15').astype('M8[ns]')],
        "forecast_time": [np.datetime64('2020-01-01T00:00').astype('M8[ns]'),
                          np.datetime64('2020-01-01T00:15').astype('M8[ns]')]}

    # hi res changes by 0 K, C48 changes by 1 K
    T_da = xr.DataArray(
        [[[[273., 274.], [273., 274.]]]],
        dims=["grid_yt", "grid_xt", "initialization_time", "forecast_time", ],
        coords=centered_coords
    )

    u_da = xr.DataArray(
        [[[[20, 30], [20., 30.]]], [[[30, 40], [30., 40.]]]],
        dims=["grid_y", "grid_xt", "initialization_time", "forecast_time"],
        coords=x_component_edge_coords)

    ds = xr.Dataset(
        {
            "T": T_da,
            "sphum": T_da,
            "u": u_da,
            "v": u_da
        }
    )
    return ds


def test__create_train_cols(test_training_raw_ds):
    train_ds = _create_train_cols(test_training_raw_ds)
    assert train_ds.Q1.values == -1./(15.*60)
    assert train_ds.QU.values == -10./(15.*60)


