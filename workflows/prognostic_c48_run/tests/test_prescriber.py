from runtime.steppers.prescriber import PrescriberConfig, Prescriber
from fv3gfs.util.testing import DummyComm
import fv3gfs.util
import numpy as np
import xarray as xr
import cftime
import pytest


@pytest.fixture(scope="module")
def external_dataset_path(tmpdir_factory):
    nxy = 8
    ntile = 6
    time = [cftime.DatetimeJulian(2016, 8, 1, 0, 15, 0, 26)]
    coords = {
        "grid_xt": xr.DataArray(np.arange(nxy), dims=["grid_xt"]),
        "grid_yt": xr.DataArray(np.arange(nxy), dims=["grid_yt"]),
        "tile": xr.DataArray(range(6), dims=["tile"]),
        "time": time,
    }
    da = xr.DataArray(
        np.ones([ntile, len(time), nxy, nxy]),
        dims=["tile", "time", "grid_yt", "grid_xt"],
        coords=coords,
    )
    vars_ = ["DSWRFsfc_coarse", "DLWRFsfc_coarse", "USWRFsfc_coarse"]
    ds = xr.Dataset({var: da for var in vars_})
    path = str(tmpdir_factory.mktemp("external_dataset.zarr"))
    ds.to_zarr(path, consolidated=True)
    return path


def get_prescriber_config(external_dataset_path):
    return PrescriberConfig(
        dataset_key=external_dataset_path,
        variables=["DSWRFsfc", "USWRFsfc", "DLWRFsfc"],
        rename={
            "DSWRFsfc": (
                "override_for_time_adjusted_total_sky_downward_shortwave_flux_at_surface"  # noqa
            ),
            "DLWRFsfc": (
                "override_for_time_adjusted_total_sky_downward_longwave_flux_at_surface"
            ),
            "NSWRFsfc": (
                "override_for_time_adjusted_total_sky_net_shortwave_flux_at_surface"
            ),
        },
    )


def get_communicator():
    layout = (1, 1)
    rank = 0
    total_ranks = 6
    shared_buffer = {}
    communicator = fv3gfs.util.CubedSphereCommunicator(
        DummyComm(rank, total_ranks, shared_buffer),
        fv3gfs.util.CubedSpherePartitioner(fv3gfs.util.TilePartitioner(layout)),
    )
    return communicator


def get_prescriber(external_dataset_path):
    prescriber = Prescriber(
        config=get_prescriber_config(external_dataset_path),
        communicator=get_communicator(),
    )
    return prescriber


def test_prescriber(external_dataset_path):
    prescriber = get_prescriber(external_dataset_path)
    time = cftime.DatetimeJulian(2016, 8, 1, 0, 15, 0)
    state = {}
    tendencies, diags, state_updates = prescriber(time, state)
    assert (
        "override_for_time_adjusted_total_sky_downward_shortwave_flux_at_surface"
        in state_updates
    )


# def get_c8_prescribed_state_list(layout, variables):
#     total_ranks = layout[0] * layout[1]
#     shared_buffer = {}
#     communicator_list = []
#     for rank in range(total_ranks):
#         communicator = fv3gfs.util.CubedSphereCommunicator(
#             DummyComm(rank, total_ranks, shared_buffer),
#             fv3gfs.util.CubedSpherePartitioner(fv3gfs.util.TilePartitioner(layout)),
#         )
#         communicator_list.append(communicator)
#     state_list = []
#     for communicator in communicator_list:
#         state_list.append(
#             fv3gfs.util.open_restart(
#                 os.path.join(DATA_DIRECTORY, "c12_restart"),
#                 communicator,
#                 only_names=only_names,
#             )
#         )
#     return
