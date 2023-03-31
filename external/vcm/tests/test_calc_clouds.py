import xarray as xr
import pytest
from vcm import gridcell_to_incloud_condensate, incloud_to_gridcell_condensate


def get_cloud_arrays():
    cloud_fraction = xr.DataArray([1.0e-3, 1.0e-2, 1.0e-1], dims=["x"])
    gridcell_mean_condensate = xr.DataArray([1.0e-3, 1.0e-3, 1.0e-3], dims=["x"])
    incloud_condensate = xr.DataArray([1.0e-2, 1.0e-2, 1.0e-2], dims=["x"])
    return xr.Dataset(
        {
            "cloud_fraction": cloud_fraction,
            "gridcell_mean_condensate": gridcell_mean_condensate,
            "incloud_condensate": incloud_condensate,
        }
    )


@pytest.mark.parametrize(
    ["climit1", "climit2", "expected_incloud"],
    [
        pytest.param(1.0e-3, 5.0e-2, [1.0e-3, 2.0e-2, 1.0e-2], id="default"),
        pytest.param(1.0e-2, 5.0e-2, [1.0e-3, 1.0e-3, 1.0e-2], id="higher_climit1"),
        pytest.param(1.0e-3, 1.0e-2, [1.0e-3, 1.0e-1, 1.0e-2], id="lower_climit2"),
    ],
)
def test_to_incloud_climits(climit1, climit2, expected_incloud):
    cloud_ds = get_cloud_arrays()
    incloud_condensate = gridcell_to_incloud_condensate(
        cloud_ds["cloud_fraction"],
        cloud_ds["gridcell_mean_condensate"],
        climit1=climit1,
        climit2=climit2,
    )
    expected_incloud = xr.DataArray(expected_incloud, dims=["x"])
    xr.testing.assert_allclose(incloud_condensate, expected_incloud)


@pytest.mark.parametrize(
    ["climit1", "climit2", "expected_gridcell"],
    [
        pytest.param(1.0e-3, 5.0e-2, [1.0e-2, 5.0e-4, 1.0e-3], id="default"),
        pytest.param(1.0e-2, 5.0e-2, [1.0e-2, 1.0e-2, 1.0e-3], id="higher_climit1"),
        pytest.param(1.0e-3, 1.0e-2, [1.0e-2, 1.0e-4, 1.0e-3], id="lower_climit2"),
    ],
)
def test_to_gridcell_climits(climit1, climit2, expected_gridcell):
    cloud_ds = get_cloud_arrays()
    gridcell_condensate = incloud_to_gridcell_condensate(
        cloud_ds["cloud_fraction"],
        cloud_ds["incloud_condensate"],
        climit1=climit1,
        climit2=climit2,
    )
    expected_gridcell = xr.DataArray(expected_gridcell, dims=["x"])
    xr.testing.assert_allclose(gridcell_condensate, expected_gridcell)


def test_condensate_roundtrip():
    cloud_ds = get_cloud_arrays()
    gridcell_condensate = incloud_to_gridcell_condensate(
        cloud_ds["cloud_fraction"], cloud_ds["incloud_condensate"],
    )
    incloud_condensate = gridcell_to_incloud_condensate(
        cloud_ds["cloud_fraction"], gridcell_condensate,
    )
    xr.testing.assert_allclose(incloud_condensate, cloud_ds["incloud_condensate"])
    incloud_condensate = gridcell_to_incloud_condensate(
        cloud_ds["cloud_fraction"], cloud_ds["gridcell_mean_condensate"],
    )
    gridcell_condensate = incloud_to_gridcell_condensate(
        cloud_ds["cloud_fraction"], incloud_condensate,
    )
    xr.testing.assert_allclose(
        gridcell_condensate, cloud_ds["gridcell_mean_condensate"]
    )
