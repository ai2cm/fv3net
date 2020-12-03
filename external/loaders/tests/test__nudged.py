import pytest
import os
import xarray as xr
import synth
from loaders import TIME_NAME
from loaders.mappers._nudged._nudged import open_nudge_to_fine, open_nudge_to_obs

NTIMES = 12


@pytest.fixture(scope="module")
def state_before_nudging_schema(datadir_module):
    nudge_data_schema = datadir_module.join(f"state_after_timestep.json")
    with open(nudge_data_schema) as f:
        schema = synth.load(f)

    data = synth.generate(schema)

    return data.isel({TIME_NAME: slice(0, NTIMES)})


@pytest.fixture(scope="module")
def nudge_to_fine_tendencies(datadir_module):
    nudge_to_fine_tendencies_schema = datadir_module.join(
        f"nudge_to_fine_tendencies.json"
    )
    with open(nudge_to_fine_tendencies_schema) as f:
        schema = synth.load(f)

    nudging_tend = synth.generate(schema)

    return nudging_tend.isel({TIME_NAME: slice(0, NTIMES)})


@pytest.fixture(scope="module")
def physics_tendencies(datadir_module):
    physics_tendencies_schema = datadir_module.join(f"physics_tendencies.json")
    with open(physics_tendencies_schema) as f:
        schema = synth.load(f)

    physics_tendencies = synth.generate(schema)

    return physics_tendencies.isel({TIME_NAME: slice(0, NTIMES)})


@pytest.fixture(scope="module")
def nudge_to_fine_data_dir(
    datadir_module,
    state_before_nudging_schema,
    physics_tendencies,
    nudge_to_fine_tendencies,
):
    all_data = {"physics_tendencies": physics_tendencies}
    all_data.update({"nudge_to_fine_tendencies": nudge_to_fine_tendencies})
    all_data.update({"state_after_timestep": state_before_nudging_schema})

    for filestem, ds in all_data.items():
        filepath = os.path.join(datadir_module, f"{filestem}.zarr")
        ds.to_zarr(filepath)

    return str(datadir_module)


NUDGING_VARIABLES = [
    "air_temperature",
    "specific_humidity",
    "x_wind",
    "y_wind",
    "pressure_thickness_of_atmospheric_layer",
]


@pytest.mark.regression
def test_open_nudge_to_fine(nudge_to_fine_data_dir):

    mapper = open_nudge_to_fine(
        nudge_to_fine_data_dir, NUDGING_VARIABLES, consolidated=False
    )

    key = list(mapper.keys())[0]
    mapper[key]["air_temperature"]
    mapper[key]["specific_humidity"]
    mapper[key]["dQ1"]
    mapper[key]["dQ2"]
    mapper[key]["dQxwind"]
    mapper[key]["dQywind"]
    mapper[key]["pQ1"]
    mapper[key]["pQ2"]


@pytest.mark.regression
@pytest.mark.parametrize(
    ["nudging_timestep_seconds", "nudging_variables"],
    [
        (900.0, NUDGING_VARIABLES),
        (60.0, NUDGING_VARIABLES),
        (900.0, NUDGING_VARIABLES[:2]),
    ],
)
def test_open_nudge_to_fine_subtract_nudging(
    nudge_to_fine_data_dir, nudging_timestep_seconds, nudging_variables
):

    nudging_variable_state = xr.open_zarr(
        os.path.join(nudge_to_fine_data_dir, "state_after_timestep.zarr"),
        consolidated=False,
    )
    nudge_to_fine_tendencies = xr.open_zarr(
        os.path.join(nudge_to_fine_data_dir, "nudge_to_fine_tendencies.zarr"),
        consolidated=False,
    )

    mapper = open_nudge_to_fine(
        nudge_to_fine_data_dir,
        nudging_variables,
        nudging_dt_seconds=nudging_timestep_seconds,
        consolidated=False,
    )

    for nudging_variable in nudging_variables:
        before_nudging_variable_state = (
            nudging_variable_state[nudging_variable]
            - nudge_to_fine_tendencies[f"{nudging_variable}_tendency_due_to_nudging"]
            * nudging_timestep_seconds
        ).isel(time=0)
        key = sorted(list(mapper.keys()))[0]
        xr.testing.assert_allclose(
            mapper[key][nudging_variable], before_nudging_variable_state
        )


@pytest.mark.regression
def test_open_nudge_to_obs(nudge_to_fine_data_dir):

    merge_files = [
        "state_after_timestep.zarr",
        "physics_tendencies.zarr",
        "nudge_to_fine_tendencies.zarr",
    ]
    rename_vars = {
        "air_temperature_tendency_due_to_nudging": "dQ1",
        "specific_humidity_tendency_due_to_nudging": "dQ2",
        "x_wind_tendency_due_to_nudging": "dQxwind",
        "y_wind_tendency_due_to_nudging": "dQywind",
        "tendency_of_air_temperature_due_to_fv3_physics": "pQ1",
        "tendency_of_specific_humidity_due_to_fv3_physics": "pQ2",
    }
    mapper = open_nudge_to_obs(
        nudge_to_fine_data_dir,
        merge_files=merge_files,
        rename_vars=rename_vars,
        consolidated=False,
    )
    key = list(mapper.keys())[0]
    mapper[key]["air_temperature"]
    mapper[key]["specific_humidity"]
    mapper[key]["dQ1"]
    mapper[key]["dQ2"]
    mapper[key]["dQxwind"]
    mapper[key]["dQywind"]
    mapper[key]["pQ1"]
    mapper[key]["pQ2"]
