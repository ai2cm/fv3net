import pytest
import xarray as xr
from vcm import safe
from loaders.mappers._fine_resolution_budget import FineResolutionSources


budget_ds = xr.Dataset(
    dict(
        air_temperature=xr.DataArray(
            [270.0], [(["x"], [1.0])], ["x"], attrs={"units": "K"}
        ),
        air_temperature_physics=xr.DataArray(
            [0.1], [(["x"], [1.0])], ["x"], attrs={"units": "K/s"}
        ),
        air_temperature_saturation_adjustment=xr.DataArray(
            [0.2], [(["x"], [1.0])], ["x"], attrs={"units": "K/s"}
        ),
        air_temperature_convergence=xr.DataArray(
            [-0.1], [(["x"], [1.0])], ["x"], attrs={"units": "K/s"}
        ),
        specific_humidity=xr.DataArray(
            [1.0e-3], [(["x"], [1.0])], ["x"], attrs={"units": "kg/kg"}
        ),
        specific_humidity_physics=xr.DataArray(
            [1.0e-6], [(["x"], [1.0])], ["x"], attrs={"units": "kg/kg/s"}
        ),
        specific_humidity_saturation_adjustment=xr.DataArray(
            [2.0e-6], [(["x"], [1.0])], ["x"], attrs={"units": "kg/kg/s"}
        ),
        specific_humidity_convergence=xr.DataArray(
            [-1.0e-6], [(["x"], [1.0])], ["x"], attrs={"units": "kg/kg/s"}
        ),
    )
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


@pytest.mark.parametrize(
    ["offset", "key", "expected"],
    [
        pytest.param(0, "20160901.001500", "20160901.001500", id="zero offset"),
        pytest.param(60, "20160901.001500", "20160901.001600", id="non-zero offset"),
    ],
)
def test__timestamp_key_to_midpoint(offset, key, expected, fine_res_mapper):
    fine_res_source_mapper = FineResolutionSources(fine_res_mapper, offset)
    midpoint_time = fine_res_source_mapper._timestamp_key_to_midpoint(key, offset)
    assert midpoint_time == expected


@pytest.mark.parametrize(
    ["offset", "midpoint_time", "expected"],
    [
        pytest.param(0, "20160901.001500", "20160901.001500", id="zero offset"),
        pytest.param(60, "20160901.001500", "20160901.001400", id="non-zero offset"),
    ],
)
def test__midpoint_to_timestamp_key(offset, midpoint_time, expected, fine_res_mapper):
    fine_res_source_mapper = FineResolutionSources(fine_res_mapper, offset)
    key = fine_res_source_mapper._midpoint_to_timestamp_key(midpoint_time, offset)
    assert key == expected


def test_FineResolutionSources(fine_res_mapper):
    fine_res_source_mapper = FineResolutionSources(fine_res_mapper)
    source_ds = fine_res_source_mapper["20160901.001500"]
    safe.get_variables(
        source_ds, ["dQ1", "dQ2", "pQ1", "pQ2", "air_temperature", "specific_humidity"]
    )
