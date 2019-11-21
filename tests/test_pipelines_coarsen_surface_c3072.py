import apache_beam as beam
import numpy as np
import xarray as xr
from apache_beam.testing.test_pipeline import TestPipeline

from fv3net.pipelines import coarsen_surface_c3072, common
from vcm import cubedsphere


def _test_data():
    n = 6
    x = np.arange(n) + 1
    da = xr.DataArray(x, dims=["x"], name="a", coords={"x": x})
    yield "a", da.isel(x=slice(0, n // 2))
    yield "a", da.isel(x=slice(n // 2 - 1, None))


def data_from_disk():
    suffix = "tileX.nc.0000"
    prefix = "PRATEsfc/gfsphysics_15min_fine_PRATEsfc"
    for fname in cubedsphere.all_filenames(prefix):
        tile = int(fname[-9])
        yield {"name": "PRATEsfc", "tile": tile}, xr.open_dataset(fname)


def mock_data():
    for tile in range(6):
        for name, ds in _test_data():
            yield {"name": name, "tile": tile}, ds.to_dataset()


def test_CombineSubtilesByKey(tmpdir):
    def _name(key):
        return "gs://vcm-ml-data/TESTDELETE/{name}.tile{tile}.nc".format(**key)

    with TestPipeline() as p:
        data = (
            p
            | beam.Create(mock_data())
            | common.CombineSubtilesByKey()
            | common.WriteToNetCDFs(_name)
        )


def test__name():
    key = {"tile": 3, "name": "a"}
    path = coarsen_surface_c3072._name(key)
    assert (
        path
        == f"gs://vcm-ml-data/2019-11-06-X-SHiELD-gfsphysics-diagnostics-coarsened/C384/a.tile3.nc"
    )
