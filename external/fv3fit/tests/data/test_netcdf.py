from pathlib import Path

import fv3fit.data
import numpy as np
import vcm


def test_NCDirLoader(tmp_path: Path):
    path = tmp_path / "a.nc"
    cache = tmp_path / ".cache"
    ds = vcm.cdl_to_dataset(
        """
    netcdf A {
        dimensions:
            sample = 4;
            z = 1;
        variables:
            float a(sample, z);
        data:
            a = 0, 1, 2, 3;
    }
    """
    )

    ds.to_netcdf(path.as_posix())
    loader = fv3fit.data.NCDirLoader(tmp_path.as_posix())
    tfds = loader.open_tfdataset(cache.as_posix(), ["a"])
    for data in tfds.as_numpy_iterator():
        a = data["a"]
    np.testing.assert_array_equal(ds["a"], a)


def test_Netcdf_from_dict():
    fv3fit.data.NCDirLoader.from_dict({"url": "some/path"})
