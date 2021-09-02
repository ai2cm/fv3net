import pytest
import cftime
import numpy as np
import xarray as xr
from datetime import timedelta

import loaders.mappers._emulation as emu


def save_training_data(start_times, tmpdir):

    filenames = ["physics_tendencies.zarr", "state_after_timestep.zarr"]
    tdelta = timedelta(hours=1)

    # generate 3 1-hr snapshots as a dataset and save
    for start in start_times:
        ds_times = [start + tdelta * i for i in range(3)]
        curr_time_str = start.strftime("%Y%m%d.%H%M%S")
        for name in filenames:
            ds = xr.Dataset(
                {
                    name[0:5]: (
                        ["time", "tile", "z", "y", "x"],
                        np.random.randn(len(ds_times), 2, 3, 4, 5),
                    )
                },
                coords={"time": ds_times},
            )
            ds.to_zarr(str(tmpdir.join(curr_time_str, name)), consolidated=True)


def test_open_phys_emu_training(tmpdir):
    init_times = [
        cftime.DatetimeJulian(2016, 8, 1),
        cftime.DatetimeJulian(2016, 8, 1, 4),
    ]
    init_time_strs = [t.strftime("%Y%m%d.%H%M%S") for t in init_times]
    save_training_data(init_times, tmpdir)

    mapper = emu.open_phys_emu_training(tmpdir, init_time_strs)
    assert len(mapper) == 6  # 2x 3-hr datasets combined


@pytest.mark.xfail
def test_open_phys_emu_training_time_overlap_error(tmpdir):
    init_times = [
        cftime.DatetimeJulian(2016, 8, 1),
        cftime.DatetimeJulian(2016, 8, 1, 2),
    ]
    init_time_strs = [t.strftime("%Y%m%d.%H%M%S") for t in init_times]
    save_training_data(init_times, tmpdir)

    with pytest.raises(xr.core.merge.MergeError):
        emu.open_phys_emu_training(tmpdir, init_time_strs)
