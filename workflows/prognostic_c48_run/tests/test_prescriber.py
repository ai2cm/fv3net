from runtime.steppers.prescriber import PrescriberConfig, Prescriber, get_timesteps, _get_time_interval_seconds
from fv3gfs.util.testing import DummyComm
import fv3gfs.util
import numpy as np
import xarray as xr
import cftime
import pytest

init_time = cftime.DatetimeJulian(2016, 8, 1, 0, 15, 0)
time_2 = cftime.DatetimeJulian(2016, 8, 1, 0, 30, 0)
time_3 = cftime.DatetimeJulian(2016, 8, 1, 0, 45, 0)
time_4 = cftime.DatetimeJulian(2016, 8, 1, 1, 0, 0)
time_5 = cftime.DatetimeJulian(2016, 8, 1, 1, 15, 0)

NXY = 8
NTILE = 6
TIME_COORD = [cftime.DatetimeJulian(2016, 8, 1, 0, 15, 0)]


@pytest.mark.parametrize(
    ["init_time", "timestep_seconds", "n_timesteps", "expected"],
    [
        pytest.param(init_time, 900.0, 2, [time_2, time_3], id="base"),
        pytest.param(time_2, 900.0, 2, [time_3, time_4], id="init_later"),
        pytest.param(init_time, 1800.0, 2, [time_3, time_5], id="long_timestep"),
        pytest.param(init_time, 900.0, 3, [time_2, time_3, time_4], id="3_timesteps"),
    ],
)
def test_get_timesteps(init_time, timestep_seconds, n_timesteps, expected):
    timesteps = get_timesteps(init_time, timestep_seconds, n_timesteps)
    assert timesteps == expected


def get_dataset(vars_, sizes, time_coord):
    coords = {
        dim: xr.DataArray(np.arange(size), dims=[dim]) for dim, size in sizes.items()
    }
    coords["tile"] = xr.DataArray(range(NTILE), dims=["tile"])
    coords["time"] = xr.DataArray(time_coord, dims=["time"])
    x_dim = [key for key in sizes.keys() if "x" in key][0]
    y_dim = [key for key in sizes.keys() if "y" in key][0]
    ds = xr.Dataset(
        {
            var: get_dataarray(
                (y_dim, sizes[y_dim]), (x_dim, sizes[x_dim]), value, coords
            )
            for var, value in vars_.items()
        }
    )
    return ds


def get_dataarray(y, x, value, coords):
    da = xr.DataArray(
        np.full([NTILE, 1, y[1], x[1]], value),
        dims=["tile", "time", y[0], x[0]],
        coords=coords,
    )
    da.attrs["units"] = "some_units"
    return da


@pytest.fixture(scope="module")
def external_dataset_path(tmpdir_factory):
    vars_ = {
        "override_for_time_adjusted_total_sky_downward_shortwave_flux_at_surface": 10.0,
        "override_for_time_adjusted_total_sky_downward_longwave_flux_at_surface": 5.0,
        "override_for_time_adjusted_total_sky_net_shortwave_flux_at_surface": 8.0,
    }
    sizes = {"y": NXY, "x": NXY}
    ds = get_dataset(vars_, sizes, TIME_COORD)
    path = str(tmpdir_factory.mktemp("external_dataset.zarr"))
    ds.to_zarr(path, consolidated=True)
    return path


def get_prescriber_config(external_dataset_path):
    return PrescriberConfig(
        dataset_key=external_dataset_path,
        variables=[
            "override_for_time_adjusted_total_sky_downward_shortwave_flux_at_surface",
            "override_for_time_adjusted_total_sky_net_shortwave_flux_at_surface",
            "override_for_time_adjusted_total_sky_downward_longwave_flux_at_surface",
        ],
    )


@pytest.fixture(params=[(1, 1), (2, 2)], scope="module")
def layout(request):
    return request.param


def get_communicators(layout):
    rank = 0
    total_ranks = 6 * layout[0] * layout[1]
    shared_buffer = {}
    communicator_list = []
    for rank in range(total_ranks):
        communicator = fv3gfs.util.CubedSphereCommunicator(
            DummyComm(rank, total_ranks, shared_buffer),
            fv3gfs.util.CubedSpherePartitioner(fv3gfs.util.TilePartitioner(layout)),
        )
        communicator_list.append(communicator)
    return communicator_list


def get_prescribers(external_dataset_path, layout):
    communicator_list = get_communicators(layout)
    prescriber_list = []
    for communicator in communicator_list:
        prescriber = Prescriber(
            config=get_prescriber_config(external_dataset_path),
            communicator=communicator,
        )
        prescriber_list.append(prescriber)
    return prescriber_list


@pytest.fixture(scope="module")
def prescriber_output(external_dataset_path, layout):
    prescriber_list = get_prescribers(external_dataset_path, layout)
    state_updates_list, tendencies_list = [], []
    for prescriber in prescriber_list:
        tendencies, _, state_updates = prescriber(TIME_COORD[0], {})
        state_updates_list.append(state_updates)
        tendencies_list.append(tendencies)
    return state_updates_list, tendencies_list


def get_expected_state_updates(layout):
    vars_ = {
        "override_for_time_adjusted_total_sky_downward_shortwave_flux_at_surface": 10.0,
        "override_for_time_adjusted_total_sky_downward_longwave_flux_at_surface": 5.0,
        "override_for_time_adjusted_total_sky_net_shortwave_flux_at_surface": 8.0,
    }
    sizes = {"y": NXY // layout[0], "x": NXY // layout[1]}
    ds = get_dataset(vars_, sizes, TIME_COORD)
    ds = ds.sel(time=TIME_COORD[0], tile=0).drop_vars(["tile", "y", "x"])
    state_updates = {name: ds[name] for name in ds.data_vars}
    return state_updates


def test_prescribed_state_updates(layout, prescriber_output):
    expected = get_expected_state_updates(layout)
    state_updates_list = prescriber_output[0]
    for state_updates in state_updates_list:
        assert set(state_updates.keys()) == set(expected.keys())
        for name in expected.keys():
            xr.testing.assert_allclose(expected[name], state_updates[name])
            assert "units" in state_updates[name].attrs


def test_no_tendencies(prescriber_output):
    tendencies_list = prescriber_output[1]
    for tendencies in tendencies_list:
        assert not tendencies


def test__get_time_interval_seconds():
    times = [init_time, time_2]
    time_coord = xr.DataArray(times, dims=["time"], coords={"time": times})
    assert _get_time_interval_seconds(time_coord) == 900.