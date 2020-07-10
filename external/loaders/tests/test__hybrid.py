from loaders.mappers import open_fine_resolution_nudging_hybrid
import synth
import numpy as np


def test_open_fine_resolution_nudging_hybrid(tmpdir):
    nudging_url = str(tmpdir.mkdir("nudging"))
    fine_url = str(tmpdir.mkdir("fine_res"))

    # timestep info
    timestep1 = "20160801.000730"
    timestep1_end = "20160801.001500"
    timestep1_npdatetime_fmt = "2016-08-01T00:15:00"
    timestep2 = "20160801.002230"
    timestep2_npdatetime_fmt = "2016-08-01T00:30:00"

    times_numpy = [
        np.datetime64(timestep1_npdatetime_fmt),
        np.datetime64(timestep2_npdatetime_fmt),
    ]
    times_centered_str = [timestep1, timestep2]

    # generate dataset
    synth.generate_nudging(nudging_url, times_numpy)
    synth.generate_fine_res(fine_url, times_centered_str)

    # test opener
    data = open_fine_resolution_nudging_hybrid(
        None, {"nudging_url": nudging_url}, {"fine_res_url": fine_url}
    )
    data[timestep1_end]
