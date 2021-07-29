import xarray as xr
import pytest
import fv3fit
from vcm import safe
import numpy as np
import tempfile
from fv3fit._shared import stack_non_vertical


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


@pytest.fixture
def nz():
    return 4


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
    ds_pred = predictor.predict(gridded_dataset)
    assert sorted(list(ds_pred.data_vars.keys())) == sorted(output_variables)

    for name in output_variables:
        assert np.all(
            stack_non_vertical(ds_pred[name]).values == outputs[name][None, :]
        )
        assert stack_non_vertical(ds_pred[name]).values.shape[0] == len(
            ds_stacked["sample"]
        )


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
        fv3fit.dump(predictor, tempdir)
        predictor = fv3fit.load(tempdir)

    ds_pred = predictor.predict(gridded_dataset)

    assert sorted(list(ds_pred.data_vars.keys())) == sorted(output_variables)
    for name in output_variables:
        assert np.all(
            stack_non_vertical(ds_pred[name]).values == outputs[name][None, :]
        )
