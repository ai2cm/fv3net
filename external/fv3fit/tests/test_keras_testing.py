import xarray as xr
import pytest
from fv3fit.keras import DummyModel
import fv3fit
from vcm import safe
import numpy as np
import tempfile


def get_gridded_dataset(nz):
    zdim, ydim, xdim = nz, 10, 10
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


@pytest.fixture
def nz():
    return 4


@pytest.mark.parametrize(
    "dummy_model",
    [
        pytest.param((["feature0", "feature1"], ["pred0"]), id="2_1"),
        pytest.param((["feature0", "feature1"], ["pred0", "pred1"]), id="2_2"),
        pytest.param((["feature0"], ["pred0", "pred1"]), id="1_2"),
    ],
    indirect=True,
)
def test_dummy_model(dummy_model, nz):
    gridded_dataset = get_gridded_dataset(nz)

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


@pytest.mark.parametrize(
    "input_variables, output_variables",
    [
        pytest.param(["feature0", "feature1"], ["pred0"], id="2_1"),
        pytest.param(["feature0", "feature1"], ["pred0", "pred1"], id="2_2"),
        pytest.param(["feature0"], ["pred0", "pred1"], id="1_2"),
    ],
)
def test_constant_model_predict(input_variables, output_variables, nz):
    gridded_dataset = get_gridded_dataset(nz)
    outputs = get_first_columns(gridded_dataset, output_variables)
    predictor = get_predictor(input_variables, output_variables, outputs)
    ds_stacked = safe.stack_once(
        gridded_dataset, "sample", [dim for dim in gridded_dataset.dims if dim != "z"]
    ).transpose("sample", "z")

    ds_pred = predictor.predict(ds_stacked)

    assert sorted(list(ds_pred.data_vars.keys())) == sorted(output_variables)
    for name in output_variables:
        assert np.all(ds_pred[name].values == outputs[name][None, :])
        assert ds_pred[name].shape[0] == len(ds_stacked["sample"])


@pytest.mark.parametrize(
    "input_variables, output_variables",
    [
        pytest.param(["feature0", "feature1"], ["pred0"], id="2_1"),
        pytest.param(["feature0", "feature1"], ["pred0", "pred1"], id="2_2"),
        pytest.param(["feature0"], ["pred0", "pred1"], id="1_2"),
    ],
)
def test_constant_model_predict_columnwise(input_variables, output_variables, nz):
    gridded_dataset = get_gridded_dataset(nz)
    outputs = get_first_columns(gridded_dataset, output_variables)
    predictor = get_predictor(input_variables, output_variables, outputs)

    ds_pred = predictor.predict_columnwise(gridded_dataset, feature_dim="z")
    assert sorted(list(ds_pred.data_vars.keys())) == sorted(output_variables)
    ds_pred_stacked = safe.stack_once(
        ds_pred, "sample", [dim for dim in ds_pred.dims if dim != "z"]
    ).transpose("sample", "z")

    for name in output_variables:
        assert np.all(ds_pred_stacked[name].values == outputs[name][None, :])


def get_predictor(input_variables, output_variables, outputs):
    predictor = fv3fit.testing.ConstantOutputPredictor(
        sample_dim_name="sample",
        input_variables=input_variables,
        output_variables=output_variables,
    )
    predictor.set_outputs(**outputs)
    return predictor


def get_first_columns(ds, names):
    columns = {}
    for name in names:
        non_z_zeros = {d: 0 for d in ds[name].dims if d != "z"}
        columns[name] = ds[name].isel(**non_z_zeros).values
    return columns


@pytest.mark.parametrize(
    "input_variables, output_variables",
    [
        pytest.param(["feature0", "feature1"], ["pred0"], id="2_1"),
        pytest.param(["feature0", "feature1"], ["pred0", "pred1"], id="2_2"),
        pytest.param(["feature0"], ["pred0", "pred1"], id="1_2"),
    ],
)
def test_constant_model_predict_after_dump_and_load(
    input_variables, output_variables, nz
):
    gridded_dataset = get_gridded_dataset(nz)
    outputs = get_first_columns(gridded_dataset, output_variables)
    predictor = get_predictor(input_variables, output_variables, outputs)
    with tempfile.TemporaryDirectory() as tempdir:
        predictor.dump(tempdir)
        predictor = fv3fit.load(tempdir)

    ds_stacked = safe.stack_once(
        gridded_dataset, "sample", [dim for dim in gridded_dataset.dims if dim != "z"]
    ).transpose("sample", "z")

    ds_pred = predictor.predict(ds_stacked)

    assert sorted(list(ds_pred.data_vars.keys())) == sorted(output_variables)
    for name in output_variables:
        assert np.all(ds_pred[name].values == outputs[name][None, :])
