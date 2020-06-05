import numpy as np
import pytest
import xarray as xr
from vcm import safe
from fv3net.regression.loaders._transform import (
    shuffled,
    _get_chunk_indices,
    stack_dropnan_shuffle,
    FineResolutionSources,
)


@pytest.fixture
def test_gridded_dataset(request):
    num_nans, zdim, ydim, xdim = request.param
    coords = {"z": range(zdim), "y": range(ydim), "x": range(xdim)}
    # unique values for ease of set comparison in test
    var = xr.DataArray(
        [
            [[(100 * k) + (10 * j) + i for i in range(10)] for j in range(10)]
            for k in range(zdim)
        ],
        dims=["z", "y", "x"],
        coords=coords,
    )
    var = var.where(var >= num_nans)  # assign nan values
    return xr.Dataset({"var": var})


@pytest.mark.parametrize(
    "test_gridded_dataset", [(0, 1, 10, 10), (0, 10, 10, 10)], indirect=True,
)
def test_stack_dropnan_shuffle_dims(test_gridded_dataset):
    ds_grid = test_gridded_dataset
    rs = np.random.RandomState(seed=0)
    ds_train = stack_dropnan_shuffle(
        init_time_dim_name="initial_time", random_state=rs, ds=ds_grid
    )
    assert set(ds_train.dims) == {"sample", "z"}
    assert len(ds_train["z"]) == len(ds_grid.z)


@pytest.mark.parametrize(
    "test_gridded_dataset, num_finite_samples",
    [((0, 2, 10, 10), 100), ((10, 2, 10, 10), 90), ((110, 2, 10, 10), 0)],
    indirect=["test_gridded_dataset"],
)
def test_stack_dropnan_shuffle_samples(test_gridded_dataset, num_finite_samples):
    ds_grid = test_gridded_dataset
    nan_mask_2d = ~np.isnan(
        ds_grid["var"].sum("z", skipna=False)
    )  # mask if any z coord has nan
    flattened = ds_grid["var"].where(nan_mask_2d).values.flatten()
    finite_samples = flattened[~np.isnan(flattened)]
    rs = np.random.RandomState(seed=0)

    if num_finite_samples == 0:
        with pytest.raises(ValueError):
            ds_train = stack_dropnan_shuffle(
                init_time_dim_name="initial_time", random_state=rs, ds=ds_grid
            )
    else:
        ds_train = stack_dropnan_shuffle(
            init_time_dim_name="initial_time", random_state=rs, ds=ds_grid
        )
        assert len(ds_train["sample"]) == num_finite_samples
        assert set(ds_train["var"].values.flatten()) == set(finite_samples)


def test__get_chunk_indices():
    chunks = (2, 3)
    expected = [[0, 1], [2, 3, 4]]
    ans = _get_chunk_indices(chunks)
    assert ans == expected


def _stacked_dataset(sample_dim):
    m, n = 10, 2
    x = "x"
    sample = sample_dim
    return xr.Dataset(
        {"a": ([sample, x], np.ones((m, n))), "b": ([sample], np.ones((m)))},
        coords={x: np.arange(n), sample_dim: np.arange(m)},
    )


def test_shuffled():
    dataset = _stacked_dataset("sample")
    dataset.isel(sample=1)
    shuffled(dataset, "sample", np.random.RandomState(1))


def test_shuffled_dask():
    dataset = _stacked_dataset("sample").chunk()
    shuffled(dataset, "sample", np.random.RandomState(1))


air_temperature = xr.DataArray([270.0], [(["x"], [1.0])], ["x"], attrs={"units": "K"})
air_temperature_physics = xr.DataArray(
    [0.1], [(["x"], [1.0])], ["x"], attrs={"units": "K/s"}
)
air_temperature_microphysics = xr.DataArray(
    [0.2], [(["x"], [1.0])], ["x"], attrs={"units": "K/s"}
)
air_temperature_convergence = xr.DataArray(
    [-0.1], [(["x"], [1.0])], ["x"], attrs={"units": "K/s"}
)
specific_humidity = xr.DataArray(
    [1.0e-3], [(["x"], [1.0])], ["x"], attrs={"units": "kg/kg"}
)
specific_humidity_physics = xr.DataArray(
    [1.0e-6], [(["x"], [1.0])], ["x"], attrs={"units": "kg/kg/s"}
)
specific_humidity_microphysics = xr.DataArray(
    [2.0e-6], [(["x"], [1.0])], ["x"], attrs={"units": "kg/kg/s"}
)
specific_humidity_convergence = xr.DataArray(
    [-1.0e-6], [(["x"], [1.0])], ["x"], attrs={"units": "kg/kg/s"}
)
budget_ds = xr.Dataset(
    {
        "air_temperature": air_temperature,
        "air_temperature_physics": air_temperature_physics,
        "air_temperature_saturation_adjustment": air_temperature_saturation_adjustment,
        "air_temperature_convergence": air_temperature_convergence,
        "specific_humidity": specific_humidity,
        "specific_humidity_physics": specific_humidity_physics,
        "specific_humidity_saturation_adjustment": specific_humidity_microphysics,
        "specific_humidity_convergence": specific_humidity_convergence,
    }
)
apparent_source_terms = ["physics", "saturation_adjustment", "convergence"]


@pytest.mark.parametrize(
    "ds, variable_name, apparent_source_name, apparent_source_terms, expected",
    [
        pytest.param(
            budget_ds,
            "air_temperature",
            "dQ1",
            apparent_source_terms,
            budget_ds.assign(
                {
                    "dQ1": xr.DataArray(
                        [0.2],
                        [(["x"], [1.0])],
                        ["x"],
                        attrs={
                            "name": "apparent source of air_temperature",
                            "units": "K/s",
                        },
                    )
                }
            ),
            id="base case",
        ),
        pytest.param(
            budget_ds,
            "air_temperature",
            "dQ1",
            ["physics", "saturation_adjustment"],
            budget_ds.assign(
                {
                    "dQ1": xr.DataArray(
                        [0.3],
                        [(["x"], [1.0])],
                        ["x"],
                        attrs={
                            "name": "apparent source of air_temperature",
                            "units": "K/s",
                        },
                    )
                }
            ),
            id="no convergence",
        ),
        pytest.param(
            budget_ds,
            "air_temperature",
            "dQ1",
            [],
            budget_ds.assign(
                {
                    "dQ1": xr.DataArray(
                        [0.3],
                        [(["x"], [1.0])],
                        ["x"],
                        attrs={
                            "name": "apparent source of air_temperature",
                            "units": "K/s",
                        },
                    )
                }
            ),
            id="empty list",
            marks=pytest.mark.xfail,
        ),
    ],
)
def test__insert_budget_dQ(
    ds, variable_name, apparent_source_name, apparent_source_terms, expected
):
    output = FineResolutionSources._insert_budget_dQ(
        ds, variable_name, apparent_source_name, apparent_source_terms,
    )
    xr.testing.assert_allclose(output["dQ1"], expected["dQ1"])
    assert output["dQ1"].attrs == expected["dQ1"].attrs


@pytest.mark.parametrize(
    "ds, variable_name, apparent_source_name, expected",
    [
        pytest.param(
            budget_ds,
            "air_temperature",
            "pQ1",
            budget_ds.assign(
                {
                    "pQ1": xr.DataArray(
                        [0.0],
                        [(["x"], [1.0])],
                        ["x"],
                        attrs={
                            "name": "coarse-res physics tendency of air_temperature",
                            "units": "K/s",
                        },
                    )
                }
            ),
            id="base case",
        ),
        pytest.param(
            xr.Dataset(
                {"air_temperature": xr.DataArray([270.0], [(["x"], [1.0])], ["x"])}
            ),
            "air_temperature",
            "pQ1",
            budget_ds.assign(
                {
                    "pQ1": xr.DataArray(
                        [0.0],
                        [(["x"], [1.0])],
                        ["x"],
                        attrs={
                            "name": "coarse-res physics tendency of air_temperature"
                        },
                    )
                }
            ),
            id="no units",
        ),
        pytest.param(
            budget_ds,
            "air_temperature",
            "pQ1",
            budget_ds.assign(
                {
                    "pQ1": xr.DataArray(
                        [0.0],
                        [(["x"], [1.0])],
                        ["x"],
                        attrs={
                            "name": "coarse-res physics tendency of air_temperature",
                            "units": "K",
                        },
                    )
                }
            ),
            id="wrong units",
            marks=pytest.mark.xfail,
        ),
    ],
)
def test__insert_budget_pQ(ds, variable_name, apparent_source_name, expected):
    output = FineResolutionSources._insert_budget_pQ(
        ds, variable_name, apparent_source_name
    )
    xr.testing.assert_allclose(output["pQ1"], expected["pQ1"])
    assert output["pQ1"].attrs == expected["pQ1"].attrs


@pytest.fixture
def fine_res_mapper():
    return {"20160901.001500": budget_ds}


def test_FineResolutionSources(fine_res_mapper):
    fine_res_source_mapper = FineResolutionSources(fine_res_mapper)
    source_ds = fine_res_source_mapper["20160901.001500"]
    safe.get_variables(
        source_ds, ["dQ1", "dQ2", "pQ1", "pQ2", "air_temperature", "specific_humidity"]
    )
