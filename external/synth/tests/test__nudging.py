from synth import generate_nudging
import numpy as np
import os
import xarray


def test_generate_nudging(tmpdir):
    generate_nudging(
        tmpdir,
        [np.datetime64("2016-08-01T00:15:00"), np.datetime64("2016-08-01T00:30:00")],
    )

    # open all the files
    for path in [
        "after_dynamics.zarr",
        "nudging_tendencies.zarr",
        "after_physics.zarr",
    ]:
        xarray.open_zarr(os.path.join(tmpdir, path))
