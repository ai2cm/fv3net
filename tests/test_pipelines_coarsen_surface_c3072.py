import uuid
import xarray as xr
from src.pipelines import coarsen_surface_c3072
import subprocess
import contextlib
from vcm.cloud import gsutil


@contextlib.contextmanager
def tmp_remote():
    fname = uuid.uuid4()
    url = f"gs://vcm-ml-data/TESTFILEDELETEME-{fname}.nc"
    yield url
    subprocess.check_call(['gsutil', 'rm', url])


def test_upload_nc():
    with tmp_remote() as url:
        da = xr.DataArray([1.0], dims=['x'], name='a')
        coarsen_surface_c3072.upload_nc(da, url)
        assert gsutil.exists(url)

