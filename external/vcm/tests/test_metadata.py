import vcm
import xarray as xr
import numpy


def test_gfdl_to_standard_dims_correct():
    data = xr.Dataset(
        {
            "2d": (["tile", "grid_yt", "grid_xt"], numpy.ones((1, 1, 1))),
            "3d": (["tile", "pfull", "grid_yt", "grid_xt"], numpy.ones((1, 1, 1, 1))),
        }
    )

    ans = vcm.metadata.gfdl_to_standard(data)
    assert set(ans.dims) == {"tile", "x", "y", "z"}


def test_gfdl_to_standard_is_inverse_of_standard_to_gfdl():

    data = xr.Dataset(
        {
            "2d": (["tile", "grid_yt", "grid_xt"], numpy.ones((1, 1, 1))),
            "3d": (["tile", "pfull", "grid_yt", "grid_xt"], numpy.ones((1, 1, 1, 1))),
        }
    )

    back = vcm.metadata.standard_to_gfdl(vcm.metadata.gfdl_to_standard(data))
    xr.testing.assert_equal(data, back)
