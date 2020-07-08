import synth
import xarray as xr
import os


def test_generate_fine_res_correct_files(tmpdir):
    times = ["20160801.000000"]
    synth.generate_fine_res(tmpdir, times)
    os.listdir(tmpdir) == {
        "20160801.000000.tile1.nc"
        "20160801.000000.tile2.nc"
        "20160801.000000.tile3.nc"
        "20160801.000000.tile4.nc"
        "20160801.000000.tile5.nc"
        "20160801.000000.tile6.nc"
    }


def test_generate_fine_res_data_opens(tmpdir):
    times = ["20160801.000000"]
    synth.generate_fine_res(tmpdir, times)
    ds = xr.open_dataset(os.path.join(tmpdir, "20160801.000000.tile1.nc"))

    assert "tile" not in ds.dims
