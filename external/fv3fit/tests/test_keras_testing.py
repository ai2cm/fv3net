import xarray as xr
import pytest
from fv3fit.keras import DummyModel
from vcm import safe


@pytest.fixture
def gridded_dataset():
    zdim, ydim, xdim = 2, 10, 10
    coords = {"y": range(ydim), "x": range(xdim), "initial_time": [0]}
    # unique values for ease of set comparison in test
    var = xr.DataArray(
        [
            [
                [[(100 * z) + (10 * y) + x for x in range(xdim)] for y in range(ydim)]
                for z in range(zdim)
            ]
        ],
        dims=["initial_time", "z", "y", "x"],
        coords=coords,
    )
    ds = xr.Dataset(
        dict(
            **{f"feature{i}": (i + 1) * var for i in range(5)},
            **{f"pred{i}": (i + 1) * var for i in range(5)},
        )
    )
    return ds


@pytest.fixture
def dummy_model(request):
    return DummyModel("sample", request.param[0], request.param[1])


def dummy_model_func(output_array):
    return xr.zeros_like(output_array)


@pytest.mark.parametrize(
    "dummy_model",
    [
        pytest.param((["feature0", "feature1"], ["pred0"]), id="2_1"),
        pytest.param((["feature0", "feature1"], ["pred0", "pred1"]), id="2_2"),
        pytest.param((["feature0"], ["pred0", "pred1"]), id="1_2"),
    ],
    indirect=True,
)
def test_dummy_model(dummy_model, gridded_dataset):

    ds_stacked = safe.stack_once(
        gridded_dataset, "sample", [dim for dim in gridded_dataset.dims if dim != "z"]
    ).transpose("sample", "z")

    dummy_model.fit([ds_stacked])
    ds_pred = dummy_model.predict(ds_stacked)

    ds_target = xr.Dataset(
        {
            output_var: dummy_model_func(ds_stacked[output_var])
            for output_var in dummy_model.output_variables
        }
    )

    xr.testing.assert_allclose(ds_pred, ds_target)
